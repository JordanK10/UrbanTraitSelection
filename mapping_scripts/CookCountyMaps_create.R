library(tidyverse)
library(sf)
library(httr)
library(jsonlite)
library(viridis)
library(patchwork) # For arranging multiple plots side by side

setwd("buildMapsa")

# Read in census tract shapefile for all of Illinois
cook_tracts <- st_read("geo_data/tl_2020_17_tract/tl_2020_17_tract.shp")

# Filter for just Co0ok County (FIPS code 17031)
cook_tracts <- cook_tracts %>%
  filter(substr(GEOID, 1, 5) == "17031")

# Get the CRS from the tracts shapefile
tracts_crs <- st_crs(cook_tracts)

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
download_url <- "https://data.cityofchicago.org/resource/cauq-8yn6.geojson"
chicago_areas <- st_read(download_url)


# Check if download was successful
if(is.null(chicago_areas)) {
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

# Standardize CRS
cook_tracts <- st_transform(cook_tracts, 4326)
if(!is.null(chicago_areas)) {
  chicago_areas <- st_transform(chicago_areas, 4326)
}

# Read the analysis data
analysis_data <- read.csv("results/price_analysis_averaged.csv")


# Filter for tract-level data (Agg == "tr")
tract_data <- analysis_data %>%
  filter(Agg == "tr")

# Create a GEOID column for joining with the shapefile
tract_data <- tract_data %>%
  mutate(
    GEOID = paste0(UnitName)  
  )

# Join with shapefile
map_data <- cook_tracts %>%
  left_join(tract_data, by = "GEOID")

# Calculate the income growth rate
map_data <- map_data %>%
  mutate(LogGrowthRatio==LogGrowthRatio)

missing_tracts <- c("17031980100", "17031381700")


# Calculate means for the variables we'll need
LogGrowthRatio_mean <- mean(map_data$LogGrowthRatio, na.rm = TRUE)
avg_cov_term_mean <- mean(map_data$TimeAverageCov, na.rm = TRUE)
frac_pop_change_mean <- mean(map_data$FracPopChange, na.rm = TRUE)



map_data <- map_data %>%
  mutate(
    LogGrowthRatio = ifelse(GEOID %in% missing_tracts & is.na(LogGrowthRatio), 
                              LogGrowthRatio_mean, LogGrowthRatio),
    TimeAverageCov = ifelse(GEOID %in% missing_tracts & is.na(TimeAverageCov), 
                        avg_cov_term_mean, TimeAverageCov),
    FracPopChange = ifelse(GEOID %in% missing_tracts & is.na(FracPopChange), 
                           frac_pop_change_mean, FracPopChange)
  )
# Modified function to create a map with custom color scales and value capping for multiple variables
create_variable_map <- function(data, variable, title, max_limit = NULL, min_limit = NULL, custom_colors = FALSE, cap_values = FALSE) {
  # Create a working copy of the data
  working_data <- data
  
  # If capping values is requested
  if (cap_values) {
    # For FracPopChange variable
    if (variable == "FracPopChange" && (!is.null(max_limit) || !is.null(min_limit))) {
      # Replace values > max_limit with max_limit (if provided)
      if (!is.null(max_limit)) {
        working_data <- working_data %>%
          mutate(across(all_of(variable), ~ifelse(.x > max_limit, max_limit, .x)))
      }
      
      # Replace values < min_limit with min_limit (if provided)
      if (!is.null(min_limit)) {
        working_data <- working_data %>%
          mutate(across(all_of(variable), ~ifelse(.x < min_limit, min_limit, .x)))
      }
    }
    
    # For LogGrowthRatio variable
    if (variable == "LogGrowthRatio") {
      # Cap at -0.015 and 0.015
      working_data <- working_data %>%
        mutate(across(all_of(variable), ~case_when(
          .x > 0.075 ~ 0.04,
          .x < -0.075 ~ -0.04,
          TRUE ~ .x
        )))
    }
  }
  
  # Split data into NA/zero and non-zero for the variable
  na_zero_data <- working_data %>% 
    filter(is.na(get(variable)) | get(variable) == 0)
  
  values_data <- working_data %>% 
    filter(!is.na(get(variable)) & get(variable) != 0)
  
  # Create the map
  p <- ggplot() +
    theme(panel.background = element_blank()) +
    # Layer for NA or zero values
    geom_sf(data = na_zero_data,
            fill = "white", color = NA, size = 0) +
    # Layer for non-zero values
    geom_sf(data = values_data,
            aes(fill = .data[[variable]]), color = NA, size = 0) +
    # Add Chicago community areas with thick, visible boundaries
    {if(!is.null(chicago_areas)) 
      geom_sf(data = chicago_areas, fill = NA, color = "black", size = 1.2, linetype = "solid")} +
    # Color scale - using custom colors if requested, otherwise inferno
    {if (custom_colors && variable == "FracPopChange") {
      # For FracPopChange, use custom diverging color scale
      scale_fill_gradientn(
        colors = c(
          "#633673",  # Color for the minimum value
          "white",     # Color for zero
          "#E77429"    # Color for the maximum value
        ),
        # Define the color breakpoints
        values = scales::rescale(c(
          if(!is.null(min_limit)) min_limit else min(values_data[[variable]], na.rm = TRUE),  # Minimum value (blue)
          0,                                           # transition point
          if(!is.null(max_limit)) max_limit else max(values_data[[variable]], na.rm = TRUE)  # Maximum value (yellow/white)
        )),
        limits = c(
          if(!is.null(min_limit)) min_limit else NA,
          if(!is.null(max_limit)) max_limit else NA
        ),
        name = variable,
        na.value = "white",
        guide = guide_colorbar(title.position = "top", title.hjust = 0.5)
      )
    } else {
      # For all other variables, use standard inferno
      scale_fill_gradientn(
        colors = c(
          "#633673",  # Color for the minimum value
          "white",     # Color for zero
          "#E77429"    # Color for the maximum value
        ),
        # Define the color breakpoints
        values = scales::rescale(c(
          if(!is.null(min_limit)) min_limit else min(values_data[[variable]], na.rm = TRUE),  # Minimum value (blue)
          0,                                           # transition point
          if(!is.null(max_limit)) max_limit else max(values_data[[variable]], na.rm = TRUE)  # Maximum value (yellow/white)
        )),
        limits = c(
          if(!is.null(min_limit)) min_limit else NA,
          if(!is.null(max_limit)) max_limit else NA
        ),
        name = variable,
        na.value = "white",
        guide = guide_colorbar(title.position = "top", title.hjust = 0.5)
      )
    }} +
    # Labels and theming
    labs(
      title = title
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      legend.position = "right",
      panel.grid.major = element_line(color = "white")
    ) +
    xlim(88.2, 87.4)
  
  return(p)
}
# Create the three maps - all using inferno color palette
income_growth_map <- create_variable_map(
  map_data, 
  "LogGrowthRatio",
  "Income Growth Rate",
  max_limit = NULL,
  min_limit = NULL,
  custom_colors = TRUE,
  cap_values = TRUE)
income_growth_map


avg_cov_term_map <- create_variable_map(
  map_data, 
  "TimeAverageCov", 
  "Average Covariance Term",
  max_limit = .01,
  min_limit = -.01,
  custom_colors = TRUE
)
avg_cov_term_map

frac_pop_change_map <- create_variable_map(
  map_data, 
  "FracPopChange",
  "Fractional Population Change",
  max_limit = 1,         # Set max limit to 1 for color scale
  min_limit = -0.5,     # Set min limit to -0.25 for color scale
  custom_colors = TRUE,  # Enable custom colors
  cap_values = TRUE)     # Cap values outside the limits
frac_pop_change_map

# Combine the maps using patchwork
combined_map <- income_growth_map + avg_cov_term_map + frac_pop_change_map +
  plot_layout(ncol = 3) +
  plot_annotation(
    title = "Cook County Census Tracts with Chicago Community Areas Outlined",
    caption = "Data source: analysis_averaged.csv",
    theme = theme(
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5)
    )
  )

# Display the combined map
print(combined_map)

# Save the combined map
ggsave("cook_county_maps_pop_price.pdf",
       plot = combined_map, width = 18, height = 8)