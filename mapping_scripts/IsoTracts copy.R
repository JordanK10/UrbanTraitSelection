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
WORKING_DIR <- "buildMapsa"

# Apply working directory
# setwd(WORKING_DIR)
TRACT_SHAPEFILE_PATH <- "geo_data/tl_2014_17_tract/tl_2014_17_tract.shp"

#------------------------------------------------------
# 1. Download and prepare tract shapefiles
#------------------------------------------------------
# Read in census tract shapefile for all of Illinois
cat("Reading census tract shapefile...\n")
all_tracts <- st_read(TRACT_SHAPEFILE_PATH)

# Filter for just Cook County using FIPS code
cook_tracts <- all_tracts %>%
  filter(substr(GEOID, 1, 5) == COOK_COUNTY_FIPS)

# Print info about the shapefile
cat("Cook County tracts loaded:", nrow(cook_tracts), "tracts\n")
cat("Shapefile columns:", names(cook_tracts), "\n")

# Get the CRS from the tracts shapefile
tracts_crs <- st_crs(cook_tracts)

#------------------------------------------------------
# 2. Download Chicago community areas
#------------------------------------------------------
# Function to download Chicago community areas
download_chicago_areas <- function(crs_to_use) {
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
      
      # Transform to match the CRS of the tract data
      chicago_areas <- st_transform(chicago_areas, crs_to_use)
      
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
chicago_areas <- download_chicago_areas(tracts_crs)

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
      
      # Transform to match the CRS of the tract data
      chicago_areas <- st_transform(chicago_areas, tracts_crs)
    } else {
      warning("Failed to retrieve Chicago community areas from API")
    }
  }, error = function(e) {
    warning("Error accessing Chicago Data Portal API: ", e$message)
  })
}

# Print info about the community areas
if(!is.null(chicago_areas)) {
  cat("Chicago community areas loaded:", nrow(chicago_areas), "areas\n")
  cat("Community area columns:", names(chicago_areas), "\n")
}

#------------------------------------------------------
# 3. Standardize CRS for all spatial data
#------------------------------------------------------
# Standardize to a common CRS (WGS84 - EPSG:4326)
cook_tracts <- st_transform(cook_tracts, 4326)
if(!is.null(chicago_areas)) {
  chicago_areas <- st_transform(chicago_areas, 4326)
}


# Set file paths
ANALYSIS_DATA_FILE <- "output_terms/bg_tr_exported_terms.csv"  # Main analysis data
ADDITIONAL_DATA_FILE <- "output_terms/bg_tr_exported_terms.csv"  # Additional data for limits

COOK_COUNTY_FIPS <- "17031"  # FIPS code for Cook County

# Covariance Term Map Settings
COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS <- FALSE  # Set to FALSE to use manual limits
COV_GRO_TERM_USE_PERCENTILE_LIMITS <- FALSE   # When TRUE, uses 10-90th percentiles; when FALSE uses min-max
COVARIANCE_MIN_GRO <- -25  # Manual minimum bound
COVARIANCE_MAX_GRO <- 25   # Manual maximum bound
COV_GRO_TERM_AUTO_COLOR_SCHEME <- FALSE  # When TRUE, selects inferno if min > 0, custom scheme otherwise
COV_GRO_TERM_FORCE_INFERNO <- FALSE     # Force using inferno color scheme regardless of min value
COV_GRO_TERM_FORCE_CUSTOM <- TRUE      # Force using custom color scheme regardless of min value

# Covariance Term Map Settings
COV_INC_TERM_USE_DATA_DRIVEN_LIMITS <- FALSE  # Set to FALSE to use manual limits
COV_INC_TERM_USE_PERCENTILE_LIMITS <- FALSE   # When TRUE, uses 10-90th percentiles; when FALSE uses min-max
COVARIANCE_MIN_INC <- -100  # Manual minimum bound
COVARIANCE_MAX_INC <- 100   # Manual maximum bound
COV_INC_TERM_AUTO_COLOR_SCHEME <- FALSE  # When TRUE, selects inferno if min > 0, custom scheme otherwise
COV_INC_TERM_FORCE_INFERNO <- FALSE     # Force using inferno color scheme regardless of min value
COV_INC_TERM_FORCE_CUSTOM <- TRUE      # Force using custom color scheme regardless of min value


# Legacy settings for backward compatibility
USE_DATA_DRIVEN_LIMITS <- TRUE  # Global setting - individual map settings will override this
USE_PERCENTILE_LIMITS <- TRUE   # Global setting - individual map settings will override this
AUTO_SELECT_COLOR_SCHEME <- TRUE  # Global setting - individual map settings will override this

# Output file names
COVARIANCES_MAP_FILE <- "output_terms/isolated_visuals/output_maps/chi_msa_maps_covars.pdf"


#------------------------------------------------------
# 4. Load and prepare the data
#------------------------------------------------------
# Read the analysis data
cat("Reading analysis data from", ANALYSIS_DATA_FILE, "\n")
analysis_data <- read.csv(ANALYSIS_DATA_FILE)


# Print info about tract data
cat("Tract level data rows:", nrow(analysis_data), "\n")
if(nrow(analysis_data) > 0) {
  cat("Tract level variables:", names(analysis_data), "\n")
}

# Create a GEOID column for joining with the shapefile
analysis_data <- analysis_data %>%
  mutate(GEOID = paste0(UnitName))

#------------------------------------------------------
# 5. Join data with shapefile
#------------------------------------------------------
# Join with shapefile
map_data <- cook_tracts %>%
  left_join(analysis_data, by = "GEOID")

# Check for join success
join_success_count <- sum(!is.na(map_data$LogGrowthRateRatio))
cat("Successfully joined data for", join_success_count, "out of", nrow(cook_tracts), "tracts\n")

# Handle missing tracts - fill with mean values
missing_tracts <- c("17031980100", "17031381700")
cat("Known missing tracts that will be filled with mean values:", paste(missing_tracts, collapse=", "), "\n")

# Calculate means for the variables we'll need
avg_cov_inc_term_mean <- mean(map_data$Sel_tr_from_bg_inc_PNC_st, na.rm = TRUE)
avg_cov_gro_term_mean <- mean(map_data$Sel_tr_from_bg_pop_PNC_st, na.rm = TRUE)

# Fill missing values
#map_data <- map_data %>%
#  mutate(
#    Sel_tr_from_bg_inc_PNC_st = ifelse(GEOID %in% missing_tracts & is.na(Sel_tr_from_bg_inc_PNC_st),
#                            avg_cov_inc_term_mean, Sel_tr_from_bg_inc_PNC_st),
#    Sel_tr_from_bg_pop_PNC_st = ifelse(GEOID %in% missing_tracts & is.na(Sel_tr_from_bg_pop_PNC_st),
#                               avg_cov_gro_term_mean, Sel_tr_from_bg_pop_PNC_st),
#  )

#------------------------------------------------------
# 6. Function for data distribution analysis 
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
# 7. Analyze data distributions for optimal color scales
#------------------------------------------------------
# Additional data file for global limits
additional_data <- analysis_data



# Covariance Term limits
if(COV_INC_TERM_USE_DATA_DRIVEN_LIMITS) {
  cat("Using data-driven limits for Covariance Term\n")
  # Use map-specific percentile setting
  USE_PERCENTILE_LIMITS_TEMP <- COV_INC_TERM_USE_PERCENTILE_LIMITS
  
  cov_inc_term_limits <- analyze_data_distribution(
    map_data, 
    "Sel_tr_from_bg_inc_PNC_st", 
    additional_data, 
    "Sel_tr_from_bg_inc_PNC_st"
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
    "Sel_tr_from_bg_pop_PNC_st", 
    additional_data, 
    "Sel_tr_from_bg_pop_PNC_st"
  )
  
  # Restore global setting
  USE_PERCENTILE_LIMITS <- USE_PERCENTILE_LIMITS_TEMP
} else {
  cat("Using manual limits for Covariance Term\n")
  cov_gro_term_limits <- list(min = COVARIANCE_MIN_GRO, max = COVARIANCE_MAX_GRO)
}


# Print all the selected limits
cat("\nSelected limits for all maps:\n")
cat("Covariance Inc: min =", cov_inc_term_limits$min, ", max =", cov_inc_term_limits$max, "\n")
cat("Covariance Gro: min =", cov_gro_term_limits$min, ", max =", cov_gro_term_limits$max, "\n")

#------------------------------------------------------
# 8. Create maps with custom color scales with LaTeX notation
#------------------------------------------------------
# Modified function to create a map with custom color scales, value capping, and LaTeX labels
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
            fill = "white", color = NA, size = 0) +
    # Layer for non-zero values
    geom_sf(data = values_data,
            aes(fill = .data[[variable]]), color = NA, size = 0) +
    # Add Chicago community areas with thick, visible boundaries
    {if(!is.null(chicago_areas)) 
      geom_sf(data = chicago_areas, fill = NA, color = "black", size = 1.2, linetype = "solid")}+
    {if (!is.null(chicago_areas))
      geom_sf(
        data = chicago_areas %>%
          #filter(!community %in% c("AVONDALE", "WEST TOWN", "LOGAN SQUARE", "IRVING PARK", "LINCOLN PARK", "NEAR NORTH SIDE")),
          #filter(!community %in% c("LOOP", "NEAR WEST SIDE", "NEAR SOUTH SIDE", "LOWER WEST SIDE", "ARMOUR SQUARE", "OAKLAND", "BRIDGEPORT", "DOUGLAS","MCKINLEY PARK")),
          filter(!community %in% c("HYDE PARK", "WOODLAWN", 'KENWOOD',  "GREATER GRAND CROSSING", "WASHINGTON PARK","ENGLEWOOD","SOUTH SHORE", "WEST ENGLEWOOD")),
        fill    = "black",
        color   = "black",
        alpha=.5,
        size    = 1.2,
        linetype= "solid"
      )
    }
  
  # Apply color scale based on parameters
  if (custom_colors) {
    # Use data-distribution-aware breakpoints if optimization is requested
    if(optimize_distribution && exists("median_value")) {
      p <- p + scale_fill_gradientn(
        colors = c(
          "#633673",  # Color for the minimum value
          "#FFFFFF",    # Color for the median/middle
          "#E77429"   # Color for the maximum value
        ),
        # Define the color breakpoints using the median for better data representation
        values = scales::rescale(c(
          min_limit,
          median_value,  # Use median instead of midpoint for better data representation
          max_limit
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
          "#FFFFFF",    # Color for zero/middle
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
      )
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
    ) +
    # Set map extent - focusing on Cook County
    xlim(-88.2, -87.4)
  
  return(p)
}

#------------------------------------------------------
# 9. Create individual maps with LaTeX notation and optimized color scales
#------------------------------------------------------
# Check if all required variables exist

required_vars <- c("Sel_tr_from_bg_inc_PNC_st","Sel_tr_from_bg_pop_PNC_st" )

available_vars <- required_vars[required_vars %in% names(map_data)]

if(length(available_vars) < 3) {
  # Print which variables are missing
  missing_vars <- setdiff(required_vars, available_vars)
  warning("The following variables are missing from the data: ", paste(missing_vars, collapse = ", "))
}



Sel_tr_from_bg_pop_PNC_st_map <- create_variable_map(
  map_data, 
  "Sel_tr_from_bg_pop_PNC_st", 
  "Cumulative Population Selection Effects (% of Total Growth)",
  latex_title = expression(paste("Cumulative Population Selection Effects (% of Total Growth)")),
  latex_legend = "X",
  max_limit = cov_gro_term_limits$max,
  min_limit = cov_gro_term_limits$min,
  auto_color_scheme = COV_GRO_TERM_AUTO_COLOR_SCHEME,
  force_inferno = COV_GRO_TERM_FORCE_INFERNO,
  force_custom = COV_GRO_TERM_FORCE_CUSTOM,
  cap_values = TRUE,
  optimize_distribution = COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS
)


Sel_tr_from_bg_inc_PNC_st_map <- create_variable_map(
  map_data, 
  "Sel_tr_from_bg_inc_PNC_st", 
  "Cumulative Income Selection Effects (% of Total Growth)",
  latex_title = expression(paste("Cumulative Income Selection Effects (% of Total Growth) ")),
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
combined_map <-  Sel_tr_from_bg_pop_PNC_st_map + Sel_tr_from_bg_inc_PNC_st_map  +
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
