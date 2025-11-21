library(tidyverse)
library(sf)
library(viridis)
library(httr)
library(jsonlite)
library(patchwork) # For arranging multiple plots side by side

#------------------------------------------------------
# USER CONFIGURABLE SETTINGS - EDIT THESE VALUES TO CUSTOMIZE PLOTS
#------------------------------------------------------
# Set working directory
#WORKING_DIR <- "buildMapsa/maps"

# Apply working directory
#setwd(WORKING_DIR)


#------------------------------------------------------
# 1. Download Chicago community areas shapefile
#------------------------------------------------------
# Function to download Chicago community areas
download_chicago_areas <- function() {
  # Create temporary file to save the shapefile
  temp_dir <- tempdir()
  temp_zip <- file.path(temp_dir, "chicago_areas.zip")
  
  # Direct download from Chicago Data Portal as shapefile - community areas
  download_url <- "https://data.cityofchicago.org/api/geospatial/cauq-8yn6?method=export&format=Shapefile"
  
  tryCatch({
    # Download the file
    download.file(download_url, temp_zip, mode = "wb")
    
    # Unzip the file
    unzip(temp_zip, exdir = temp_dir)
    
    # Find shapefile
    shp_files <- list.files(temp_dir, pattern = "\\.shp$", full.names = TRUE, recursive = TRUE)
    
    if(length(shp_files) > 0) {
      # Read the shapefile
      chicago_areas <- st_read(shp_files[1], quiet = TRUE)
      
      # Transform to a standard CRS
      chicago_areas <- st_transform(chicago_areas, 4326)
      
      return(chicago_areas)
    } else {
      warning("Could not find community areas shapefile after download")
      return(NULL)
    }
  }, error = function(e) {
    warning("Error downloading Chicago community areas: ", e$message)
    return(NULL)
  })
}

# Get Chicago community areas
chicago_areas <- download_chicago_areas()

# Check if download was successful
if(is.null(chicago_areas)) {
  warning("Failed to download Chicago community areas. Using API method instead.")
  
  # Try API method as fallback
  tryCatch({
    # Access Chicago Data Portal API directly for community areas
    url <- "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
    
    # Make the API request
    response <- GET(url)
    
    # Check if request was successful
    if (http_status(response)$category == "Success") {
      # Parse GeoJSON response
      geojson <- content(response, "text", encoding = "UTF-8")
      chicago_areas <- st_read(geojson, quiet = TRUE)
      
      # Transform to standard CRS
      chicago_areas <- st_transform(chicago_areas, 4326)
    } else {
      warning("Failed to retrieve Chicago community areas from API")
    }
  }, error = function(e) {
    warning("Error accessing Chicago Data Portal API: ", e$message)
  })
}

# Print info about the shapefile
if(!is.null(chicago_areas)) {
  cat("Shapefile columns:", names(chicago_areas), "\n")
}

#------------------------------------------------------
# 2. Load and prepare the data
#------------------------------------------------------
# Read the analysis data
analysis_data <- read.csv(ANALYSIS_DATA_FILE)

# Filter for community area level data (Agg == "cm")
community_data <- analysis_data 

# Print info about community data
cat("Community level data rows:", nrow(community_data), "\n")
if(nrow(community_data) > 0) {
  cat("Community level variables:", names(community_data), "\n")
}

# Additional data file for global limits
additional_data= data
#------------------------------------------------------
# 3. Join data with shapefile - Using updated joining function
#------------------------------------------------------
# Simple function to join Chicago community data with shapefile
# Focusing on community in chicago_areas and UnitName in community_data
join_chicago_community_data <- function(comm_data, shape_data) {
  # Display info about both datasets for debugging
  cat("Shape data community column values (first few):\n")
  print(head(shape_data$community))
  
  cat("\nCommunity data UnitName column values (first few):\n")
  print(head(comm_data$UnitName))
  
  # Step 1: Standardize both column values for better matching
  # Convert to uppercase and trim whitespace
  shape_data$std_community <- toupper(trimws(as.character(shape_data$community)))
  comm_data$std_unitname <- toupper(trimws(as.character(comm_data$UnitName)))
  
  # Step 2: Check for direct matches first
  # Create empty columns in each dataset for the other's values
  shape_data$matching_unitname <- NA_character_
  comm_data$matching_community <- NA_character_
  
  # Check for exact matches
  for(i in 1:nrow(shape_data)) {
    exact_match <- which(comm_data$std_unitname == shape_data$std_community[i])
    if(length(exact_match) > 0) {
      shape_data$matching_unitname[i] <- comm_data$UnitName[exact_match[1]]
    }
  }
  
  # Count how many exact matches we found
  exact_match_count <- sum(!is.na(shape_data$matching_unitname))
  cat("\nFound", exact_match_count, "exact matches between community and UnitName\n")
  
  # If we have a good number of exact matches, proceed with direct join
  if(exact_match_count > 0) {
    cat("Joining datasets using direct name matching...\n")
    
    # Join the datasets using the standardized columns
    joined_data <- shape_data %>%
      left_join(comm_data, by = c("std_community" = "std_unitname"))
    
    # Check if join was successful
    key_var <- setdiff(names(comm_data), c("UnitName", "std_unitname", "row_id"))[1]
    match_count <- sum(!is.na(joined_data[[key_var]]))
    
    cat("Direct join matched", match_count, "out of", nrow(shape_data), "areas\n")
    
    # If we got a good match, return this join
    if(match_count > 0.5 * nrow(shape_data)) {
      cat("Direct join was reasonably successful!\n")
      return(joined_data)
    }
  }
  
  # Step 3: Try fuzzy matching if exact matching didn't work well
  if(exact_match_count < 0.5 * nrow(shape_data)) {
    cat("\nTrying fuzzy string matching between community and UnitName...\n")
    
    # This uses the stringdist package for fuzzy matching
    # If not installed: install.packages("stringdist")
    library(stringdist)
    
    # Function to find the best match with a distance threshold
    find_best_match <- function(name, candidates, max_dist = 0.3) {
      if(name %in% candidates) return(name)
      
      dists <- stringdist(name, candidates, method = "jw")
      best_idx <- which.min(dists)
      
      if(dists[best_idx] <= max_dist) {
        return(candidates[best_idx])
      } else {
        return(NA_character_)
      }
    }
    
    # Create a mapping table
    mapping <- data.frame(
      community = shape_data$std_community,
      best_unitname_match = NA_character_,
      stringsAsFactors = FALSE
    )
    
    # Find best UnitName match for each community
    for(i in 1:nrow(mapping)) {
      if(is.na(shape_data$matching_unitname[i])) { # Only for non-exact matches
        mapping$best_unitname_match[i] <- find_best_match(
          mapping$community[i], 
          comm_data$std_unitname
        )
      } else {
        mapping$best_unitname_match[i] <- shape_data$std_community[i] # Keep exact matches
      }
    }
    
    # Print out the fuzzy mapping for review
    cat("\nFuzzy mapping between community and UnitName (first 10 rows):\n")
    print(head(mapping, 10))
    
    # Join using the fuzzy mapping
    shape_with_mapping <- shape_data %>%
      left_join(mapping, by = c("std_community" = "community"))
    
    joined_data <- shape_with_mapping %>%
      left_join(comm_data, by = c("best_unitname_match" = "std_unitname"))
    
    # Check how successful the fuzzy join was
    key_var <- setdiff(names(comm_data), c("UnitName", "std_unitname", "row_id"))[1]
    match_count <- sum(!is.na(joined_data[[key_var]]))
    
    cat("\nFuzzy join matched", match_count, "out of", nrow(shape_data), "areas\n")
    
    # If we got a good match, return this join
    if(match_count > 0.5 * nrow(shape_data)) {
      cat("Fuzzy join was successful!\n")
      return(joined_data)
    }
  }
  
  # -------------------------------------------------
  # Step 4: Try a simpler approach - direct join on row order
  # -------------------------------------------------
  cat("\nTrying simple join based on row order as last resort...\n")
  cat("WARNING: This assumes the rows in both datasets are in the same order!\n")
  
  # Check if datasets have similar number of rows
  if(nrow(shape_data) == nrow(comm_data)) {
    cat("Datasets have the same number of rows, proceeding with row-based join\n")
    
    # Add row numbers and join
    shape_data$row_id <- 1:nrow(shape_data)
    comm_data$row_id <- 1:nrow(comm_data)
    
    joined_data <- shape_data %>%
      left_join(comm_data, by = "row_id")
    
    return(joined_data)
  } else {
    # If datasets have different sizes, match as many as possible
    min_rows <- min(nrow(shape_data), nrow(comm_data))
    cat("Datasets have different sizes, joining first", min_rows, "rows\n")
    
    # Create row IDs only up to the minimum number of rows
    shape_data$row_id <- 1:nrow(shape_data)
    comm_data$row_id <- 1:nrow(comm_data)
    
    # Filter and join
    shape_subset <- shape_data %>% filter(row_id <= min_rows)
    comm_subset <- comm_data %>% filter(row_id <= min_rows)
    
    joined_data <- shape_subset %>%
      left_join(comm_subset, by = "row_id")
    
    return(joined_data)
  }
  
  # If all else fails
  cat("\nCould not join the datasets effectively. Please verify the data.\n")
  return(NULL)
}

# Try to join the data
if(!is.null(chicago_areas) && nrow(community_data) > 0) {
  map_data <- join_chicago_community_data(community_data, chicago_areas)
} else {
  stop("Cannot proceed without both community data and shapefile")
}

# Set file paths
ANALYSIS_DATA_FILE <- "output_terms/bg_cm_exported_terms.csv"  # Main analysis data
ADDITIONAL_DATA_FILE <- "output_terms/bg_cm_exported_terms.csv"  # Additional data for limits


# Covariance Term Map Settings
COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS <- FALSE  # Set to FALSE to use manual limits
COV_GRO_TERM_USE_PERCENTILE_LIMITS <- FALSE   # When TRUE, uses 10-90th percentiles; when FALSE uses min-max
COVARIANCE_MIN_GRO <- -40  # Manual minimum bound
COVARIANCE_MAX_GRO <- 25  # Manual maximum bound
COV_GRO_TERM_AUTO_COLOR_SCHEME <- TRUE  # When TRUE, selects inferno if min > 0, custom scheme otherwise
COV_GRO_TERM_FORCE_INFERNO <- FALSE     # Force using inferno color scheme regardless of min value
COV_GRO_TERM_FORCE_CUSTOM <- TRUE      # Force using custom color scheme regardless of min value

# Covariance Term Map Settings
COV_INC_TERM_USE_DATA_DRIVEN_LIMITS <- FALSE  # Set to FALSE to use manual limits
COV_INC_TERM_USE_PERCENTILE_LIMITS <- FALSE   # When TRUE, uses 10-90th percentiles; when FALSE uses min-max
COVARIANCE_MIN_INC <- -40  # Manual minimum bound
COVARIANCE_MAX_INC <- 30   # Manual maximum bound
COV_INC_TERM_AUTO_COLOR_SCHEME <- TRUE  # When TRUE, selects inferno if min > 0, custom scheme otherwise
COV_INC_TERM_FORCE_INFERNO <- FALSE     # Force using inferno color scheme regardless of min value
COV_INC_TERM_FORCE_CUSTOM <- TRUE      # Force using custom color scheme regardless of min value


# Legacy settings for backward compatibility
USE_DATA_DRIVEN_LIMITS <- FALSE  # Global setting - individual map settings will override this
USE_PERCENTILE_LIMITS <- FALSE   # Global setting - individual map settings will override this
AUTO_SELECT_COLOR_SCHEME <- TRUE  # Global setting - individual map settings will override this

# Output file names
COVARIANCES_MAP_FILE <- "iso_covar_comm_bg.pdf"


#------------------------------------------------------
# 4. Function for data distribution analysis 
#------------------------------------------------------
# Function to analyze the data distribution and suggest appropriate limits
analyze_data_distribution <- function(data, variable, additional_data = NULL, additional_var = NULL) {
  if(variable %in% names(data)) {
    values <- data[[variable]]
    values <- values[!is.na(values) & values != 0]
    
    # Print summary statistics
    cat("Summary statistics for", variable, ":\n")
    summary_stats <- summary(values)
    print(summary_stats)
    
    # Use percentiles or min/max based on settings
    if(USE_PERCENTILE_LIMITS) {
      # Print quantiles
      cat("\nQuantiles (10% intervals):\n")
      quantiles <- quantile(values, probs = seq(0, 1, 0.1))
      print(quantiles)
      
      # Return suggested min and max limits based on data distribution
      suggested_min <- quantiles["10%"]
      suggested_max <- quantiles["90%"]
    } else {
      # Use min/max instead of percentiles
      suggested_min <- min(values, na.rm = TRUE)
      suggested_max <- max(values, na.rm = TRUE)
    }
    
    # Check if we have additional data to consider for global limits
    if(!is.null(additional_data) && !is.null(additional_var) && additional_var %in% names(additional_data)) {
      additional_values <- additional_data[[additional_var]]
      additional_values <- additional_values[!is.na(additional_values) & additional_values != 0]
      
      if(length(additional_values) > 0) {
        cat("\nAdditional data statistics for", additional_var, ":\n")
        print(summary(additional_values))
        
        # Consider global min/max
        global_min <- min(additional_values, na.rm = TRUE)
        global_max <- max(additional_values, na.rm = TRUE)
        
        # Use the more extreme limits
        suggested_min <- min(suggested_min, global_min)
        suggested_max <- max(suggested_max, global_max)
        
        cat("\nAdjusted limits based on additional data:", additional_var, "\n")
      }
    }
    
    cat("\nSuggested min_limit:", suggested_min, "\n")
    cat("Suggested max_limit:", suggested_max, "\n")
    
    return(list(min = suggested_min, max = suggested_max))
  } else {
    cat("Variable", variable, "not found in the data.\n")
    return(NULL)
  }
}

#------------------------------------------------------
# 5. Create maps with custom color scales with LaTeX notation and optimized data distribution
#------------------------------------------------------
# Modified function to create a map with custom color scales, value capping, LaTeX labels, and optimized data distribution
create_variable_map <- function(data, variable, title, latex_title = NULL, latex_legend = NULL,
                                max_limit = NULL, min_limit = NULL, 
                                auto_color_scheme = TRUE, force_inferno = FALSE, force_custom = FALSE,
                                cap_values = FALSE, optimize_distribution = FALSE) {
  # Create a working copy of the data
  working_data <- data
  
  # Determine color scheme
  custom_colors <- TRUE  # Default
  
  if(force_inferno) {
    # Force inferno color scheme
    custom_colors <- FALSE
    cat("Forcing inferno color scheme for", variable, "\n")
  } else if(force_custom) {
    # Force custom color scheme
    custom_colors <- TRUE
    cat("Forcing custom purple-white-orange color scheme for", variable, "\n")
  } else if(auto_color_scheme && !is.null(min_limit)) {
    # Auto-select color scheme based on min_limit
    if(min_limit > 0) {
      cat("Min limit > 0, auto-selecting inferno color scheme for", variable, "\n")
      custom_colors <- FALSE
    } else {
      cat("Min limit <= 0, using custom purple-white-orange color scheme for", variable, "\n")
      custom_colors <- TRUE
    }
  }
  
  # Analyze data distribution if optimization is requested
  if(optimize_distribution) {
    # Get data quantiles for better color distribution
    if(variable %in% names(data)) {
      values <- data[[variable]]
      values <- values[!is.na(values) & values != 0]
      
      # Use 10th and 90th percentiles if no limits are provided
      if(is.null(min_limit)) {
        min_limit <- quantile(values, 0.1)
        cat("Using 10th percentile as min_limit:", min_limit, "\n")
      }
      
      if(is.null(max_limit)) {
        max_limit <- quantile(values, 0.9)
        cat("Using 90th percentile as max_limit:", max_limit, "\n")
      }
      
      # Get the median for better distribution
      median_value <- median(values)
    }
  }
  
  # If capping values is requested
  if (cap_values) {
    if (!is.null(max_limit)) {
      working_data <- working_data %>%
        mutate(across(all_of(variable), ~ifelse(.x > max_limit, max_limit, .x)))
    }
    
    if (!is.null(min_limit)) {
      working_data <- working_data %>%
        mutate(across(all_of(variable), ~ifelse(.x < min_limit, min_limit, .x)))
    }
  }
  
  # Split data into NA/zero and non-zero for the variable
  na_data <- working_data %>% 
    filter(is.na(get(variable)) )
  
  values_data <- working_data %>% 
    filter(!is.na(get(variable)) & get(variable) != 0 | get(variable) == 0)
  # Set the title - use LaTeX if provided
  map_title <- title
  if (!is.null(latex_title)) {
    map_title <- latex_title
  }
  
  # Set the legend label - use LaTeX if provided
  legend_label <- variable
  if (!is.null(latex_legend)) {
    legend_label <- latex_legend
  }
  
  
  
  # Create the map
  p <- ggplot() +
    theme(panel.background = element_blank()) +
    # Layer for NA or zero values
    geom_sf(data = na_data,
            fill = "white", color = "black", size = 0.5) +
    # Layer for non-zero values
    geom_sf(data = values_data,
            aes(fill = .data[[variable]]), color = "black", size = 0.5)
  
  
  # Apply color scale based on parameters
  if (custom_colors) {
    # Use data-distribution-aware breakpoints if optimization is requested
    if(optimize_distribution && exists("median_value")) {
      p <- p + scale_fill_gradientn(
        colors = c(
          "#633673",  # Color for the minimum value (purple)
          "white",    # Color for zero (white)
          "#E77429"   # Color for the maximum value (orange)
        ),
        values = scales::rescale(c(
          min_limit,  # Minimum value
          0,          # Zero maps to white
          max_limit   # Maximum value
        )),
        limits = c(min_limit, max_limit),
        name = legend_label,
        na.value = "white",
        guide = guide_colorbar(title.position = "top", title.hjust = 0.5)
      )
    } else {
      # Original implementation with default breakpoints
      p <- p + scale_fill_gradientn(
        colors = c(
          "#633673",  # Color for the minimum value
          "white",    # Color for zero/middle
          "#E77429"   # Color for the maximum value
        ),
        # Define the color breakpoints
        values = scales::rescale(c(
          if(!is.null(min_limit)) min_limit else min(values_data[[variable]], na.rm = TRUE),
          0,  # Use zero or midpoint
          if(!is.null(max_limit)) max_limit else max(values_data[[variable]], na.rm = TRUE)
        )),
        limits = c(
          if(!is.null(min_limit)) min_limit else NA,
          if(!is.null(max_limit)) max_limit else NA
        ),
        name = legend_label,
        na.value = "white",
        guide = guide_colorbar(title.position = "top", title.hjust = 0.5)
      )+  # Add Chicago community areas with thick, visible boundaries
  {if(!is.null(chicago_areas)) 
    geom_sf(data = chicago_areas, fill = NA, color = "black", size = 1.2, linetype = "solid")}+
    {if (!is.null(chicago_areas))
      geom_sf(
        data = chicago_areas %>%
          filter(!community %in% c("HYDE PARK", "WOODLAWN", "GREATER GRAND CROSSING", "WASHINGTON PARK","ENGLEWOOD","SOUTH SHORE", "WEST ENGLEWOOD")),
        fill    = "black",
        color   = "black",
        alpha=.5,
        size    = 1.2,
        linetype= "solid"
      )
    }
  
    }
  } else {
    # Use viridis inferno color scheme
    p <- p + scale_fill_viridis(
      option = "inferno",
      limits = c(
        if(!is.null(min_limit)) min_limit else NA,
        if(!is.null(max_limit)) max_limit else NA
      ),
      name = legend_label,
      guide = guide_colorbar(title.position = "top", title.hjust = 0.5)
    )
  }
  # Add labels and themes
  p <- p + labs(title = map_title) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "right",
      panel.grid.major = element_line(color = "white")
    )
    # Create the map
  p <- ggplot() +
    theme(panel.background = element_blank()) +
    # Layer for NA or zero values
    geom_sf(data = na_data,
            fill = "white", color = NA, size = 0) +
    # Layer for non-zero values
    geom_sf(data = values_data,
            aes(fill = .data[[variable]]), color = NA, size = 0) +
    # Add Chicago community areas with thick, visible boundaries

  
  return(p)
}

#------------------------------------------------------
# 6. Analyze data distributions for optimal color scales
#------------------------------------------------------
# Analyze the data distributions for each variable, using additional data if available

# Covariance Term limits
if(COV_INC_TERM_USE_DATA_DRIVEN_LIMITS) {
  cat("Using data-driven limits for Covariance Term\n")
  # Use map-specific percentile setting
  USE_PERCENTILE_LIMITS_TEMP <- COV_INC_TERM_USE_PERCENTILE_LIMITS
  
  cov_term_limits <- analyze_data_distribution(
    map_data, 
    "Transmitted_Sel_tr_to_cm_inc_Ratio", 
    additional_data, 
    "Transmitted_Sel_tr_to_cm_inc_Ratio"
  )
  
  # Restore global setting
  USE_PERCENTILE_LIMITS <- USE_PERCENTILE_LIMITS_TEMP
} else {
  cat("Using manual limits for Covariance Term\n")
  cov_inc_term_limits <- list(min = COVARIANCE_MIN_INC, max = COVARIANCE_MAX_INC)
}

# Covariance Term limits
if(COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS) {
  cat("Using data-driven limits for Covariance Term\n")
  # Use map-specific percentile setting
  USE_PERCENTILE_LIMITS_TEMP <- COV_GRO_TERM_USE_PERCENTILE_LIMITS
  
  cov_gro_term_limits <- analyze_data_distribution(
    map_data, 
    "Transmitted_Sel_tr_to_cm_gro_Ratio", 
    additional_data, 
    "Transmitted_Sel_tr_to_cm_gro_Ratio"
  )
  
  # Restore global setting
  USE_PERCENTILE_LIMITS <- USE_PERCENTILE_LIMITS_TEMP
} else {
  cat("Using manual limits for Covariance Term\n")
  cov_gro_term_limits <- list(min = COVARIANCE_MIN_GRO, max = COVARIANCE_MAX_GRO)
}

# Print all the selected limits
cat("\nSelected limits for all maps:\n")
cat("Covariance gro: min =", cov_gro_term_limits$min, ", max =", cov_gro_term_limits$max, "\n")
cat("Covariance inc: min =", cov_inc_term_limits$min, ", max =", cov_inc_term_limits$max, "\n")


#------------------------------------------------------
# 7. Create individual maps with LaTeX notation and optimized color scales
#------------------------------------------------------
# Check if all required variables exist
#required_vars <- c("LogGrowthRateRatio", "TimeAverageCovUn", "FracPopChange")

#required_vars <- c("PopFracChange_Overall",   "LogGrowthRateRatio", "CovDistShift_Overall" )
required_vars <- c("Transmitted_Sel_tr_to_cm_inc_Ratio",   "Transmitted_Sel_tr_to_cm_gro_Ratio" )

available_vars <- required_vars[required_vars %in% names(map_data)]



#COV PLOTS
required_vars <- c("Transmitted_Sel_tr_to_cm_inc_Ratio", "Transmitted_Sel_tr_to_cm_gro_Ratio" )

available_vars <- required_vars[required_vars %in% names(map_data)]


# Create the three maps with LaTeX notation and optimized color scales

price_cov_term_map <- create_variable_map(
  map_data, 
  "Transmitted_Sel_tr_to_cm_gro_Ratio", 
  "Population Selection Effects, Block Groups (% of Total Growth)",
  latex_title = expression(paste("Population Selection Effects, Block Groups (% of Average Growth)")),
  latex_legend = "X",
  max_limit = cov_gro_term_limits$max,
  min_limit = cov_gro_term_limits$min,
  auto_color_scheme = COV_GRO_TERM_AUTO_COLOR_SCHEME,
  force_inferno = COV_GRO_TERM_FORCE_INFERNO,
  force_custom = COV_GRO_TERM_FORCE_CUSTOM,
  cap_values = TRUE,
  optimize_distribution = COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS
)


income_cov_term_map <- create_variable_map(
  map_data, 
  "Transmitted_Sel_tr_to_cm_inc_Ratio", 
  "Income Selection Effects, Block Groups (% of Total Growth)",
  latex_title = expression(paste("Income Selection Effects, Block Groups (% of Aggregate Growth)")),
  latex_legend = "X",
  max_limit = cov_inc_term_limits$max,
  min_limit = cov_inc_term_limits$min,
  auto_color_scheme = COV_INC_TERM_AUTO_COLOR_SCHEME,
  force_inferno = COV_INC_TERM_FORCE_INFERNO,
  force_custom = COV_INC_TERM_FORCE_CUSTOM,
  cap_values = TRUE,
  optimize_distribution = COV_INC_TERM_USE_DATA_DRIVEN_LIMITS
)
#------------------------------------------------------
# 8. Create combined map
#------------------------------------------------------
# Combine the maps using patchwork
combined_map <-  income_cov_term_map + price_cov_term_map  +
  plot_layout(ncol = 2, nrow = 1) +
  plot_annotation(
    title = "Chicago Community Areas",
    caption = "Data source: American Community Survey (ACS)",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5)
    )
  )

# Display the combined map
print(combined_map)

# Save the combined map
ggsave(COVARIANCES_MAP_FILE, plot = combined_map, width = 18, height = 8)