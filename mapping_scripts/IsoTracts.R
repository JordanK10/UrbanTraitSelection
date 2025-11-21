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
setwd(WORKING_DIR)
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
ANALYSIS_DATA_FILE <- "bg_tr_exported_terms.csv"  # Main analysis data
ADDITIONAL_DATA_FILE <- "bg_tr_exported_terms.csv"  # Additional data for limits

COOK_COUNTY_FIPS <- "17031"  # FIPS code for Cook County

# Covariance Term Map Settings
COV_GRO_TERM_USE_DATA_DRIVEN_LIMITS <- FALSE  # Set to FALSE to use manual limits
COV_GRO_TERM_USE_PERCENTILE_LIMITS <- FALSE   # When TRUE, uses 10-90th percentiles; when FALSE uses min-max
COVARIANCE_MIN_GRO <- 10  # Manual minimum bound
COVARIANCE_MAX_GRO <- 12.5   # Manual maximum bound
COV_GRO_TERM_AUTO_COLOR_SCHEME <- FALSE  # When TRUE, selects inferno if min > 0, custom scheme otherwise
COV_GRO_TERM_FORCE_INFERNO <- TRUE     # Force using inferno color scheme regardless of min value
COV_GRO_TERM_FORCE_CUSTOM <- FALSE      # Force using custom color scheme regardless of min value

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
avg_cov_gro_term_mean <- mean(map_data$LogAvgIncInitial_tr, na.rm = TRUE)

# Fill missing values
#map_data <- map_data %>%
#  mutate(
#    Sel_tr_from_bg_inc_PNC_st = ifelse(GEOID %in% missing_tracts & is.na(Sel_tr_from_bg_inc_PNC_st),
#                            avg_cov_inc_term_mean, Sel_tr_from_bg_inc_PNC_st),
#    LogAvgIncInitial_tr = ifelse(GEOID %in% missing_tracts & is.na(LogAvgIncInitial_tr),
#                               avg_cov_gro_term_mean, LogAvgIncInitial_tr),
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
  