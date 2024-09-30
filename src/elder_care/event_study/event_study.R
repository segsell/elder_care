#######################
# Sub-sample Analysis #
#######################
library(foreign)
library(data.table) ## For some minor data wrangling
library(haven)     ## NB: Requires version >=0.9.0
library(fixest)
library(AER)
library(did)
library(lfe)
library(dplyr)


# Load data
data <- read.csv('/home/sebastian/Projects/elder_care/bld/event_study/sandbox.csv')
dat <- as.data.table(data, TRUE)
is.data.table(dat)

dat_female_raw <-subset(dat, gender == 2)
dat_male_raw <-subset(dat, gender == 1)


# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat_female <- dat_female_raw %>%
  filter(!distance_to_treat %in% c(-1, -3, -5, -7, 1, 3, 5, 7))
dat_male <- dat_male_raw %>%
  filter(!distance_to_treat %in% c(-1, -3, -5, 1, 3, 5))

table(dat_female$intensive_care_no_other)
table(dat_male$intensive_care_no_other)

#dat<-subset(data, time_to_treat > -15 & time_to_treat < 15)
#dat_female<-subset(dat_female, binned_distance_to_treat > -7 & binned_distance_to_treat < 7)
#dat_male<-subset(dat_male, binned_distance_to_treat > -7 & binned_distance_to_treat < 7)

to_drop <- dat_female %>%
  filter(intensive_care_general == TRUE & intensive_care_no_other == FALSE) %>%
  select(mergeid)
dat_female_drop <- dat_female %>%
  anti_join(to_drop, by = "mergeid")

dat_female_drop <- dat_female_drop %>%
  mutate(intensive_care_no_other = case_when(
    (intensive_care_general == FALSE & intensive_care_no_other == FALSE) ~ 0,
    is.na(intensive_care_general) | is.na(intensive_care_no_other) ~ 0,
    TRUE ~ intensive_care_no_other
  ))


to_drop_male <- dat_male %>%
  filter(intensive_care_general == TRUE & intensive_care_no_other == FALSE) %>%
  select(mergeid)
dat_male_drop <- dat_male %>%
  anti_join(to_drop_male, by = "mergeid")

dat_male_drop <- dat_male_drop %>%
  mutate(intensive_care_no_other = case_when(
    (intensive_care_general == FALSE & intensive_care_no_other == FALSE) ~ 0,
    is.na(intensive_care_general) | is.na(intensive_care_no_other) ~ 0,
    TRUE ~ intensive_care_no_other
  ))


x_min_value <- -10
x_max_value <- 10

y_min_value <- -0.2
y_max_value <- 700


# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat_female <- dat_female_raw %>%
  filter(!binned_distance_to_treat %in% c(-1, -3, -5, -7, -9, 1, 3, 5, 7, 9))
dat_male <- dat_male_raw %>%
  filter(!binned_distance_to_treat %in% c(-1, -3, -5, -7, -9, 1, 3, 5, 7, 9))

#######################
# Auxiliary Functions #
#######################

calculate_conf_intervals <- function(reg_summary) {
  # Extract coefficients and standard errors
  coefficients <- coef(reg_summary)
  std_errors <- reg_summary$se  # Cluster-robust standard errors (HC1)

  # Calculate lower and upper bounds for 95% confidence intervals
  lower_bound <- coefficients - 1.96 * std_errors
  upper_bound <- coefficients + 1.96 * std_errors

  # Create a data frame to store results
  results <- data.frame(
    Coefficient = coefficients,
    `95 CI Lower` = lower_bound,
    `95 CI Upper` = upper_bound
  )

  return(results)
}


extract_integer <- function(string) {
  pattern <- "(?<=::)-?\\d+(?=:)"
  extracted_integer <- as.integer(regmatches(string, regexpr(pattern, string, perl = TRUE)))
  return(extracted_integer)
}


# Function to plot event study
plot_event_study <- function(reg_summary, xlim, ylim, xlab, ylab, file_path = NULL) {
  # Calculate confidence intervals
  intervals <- calculate_conf_intervals(reg_summary)

  # Extract coefficient names (variable names)
  variable_names <- rownames(intervals)

  # Extract the integer part of the variable names
  x_ticks <- as.numeric(sapply(variable_names, extract_integer))

  # Open a graphics device if file_path is provided
  if (!is.null(file_path)) {
    png(file_path, width = 800, height = 600)
  }

  # Create a plot
  plot(x_ticks, intervals$Coefficient,
       type = 'p', pch = 20, col = 'black', cex = 1.0, # Smaller points
       xlab = xlab, ylab = ylab,
       xlim = xlim, ylim = ylim,
       yaxt = "n")  # Suppress y-axis to customize it later

  # Add error bars for 95% confidence intervals
  arrows(x_ticks, intervals$X95.CI.Lower,
         x_ticks, intervals$X95.CI.Upper,
         code = 3, angle = 90, length = 0.02, col = 'black')

  # Add y-axis with labels rotated by 90 degrees
  axis(2, las = 2)

  # Add a point at -2 with value 0 and add a vertical dashed line
  points(-2, 0, pch = 20, cex = 1.0, col = 'black')
  abline(v = -2, lty = 2, col = 'black')

  # Add a horizontal line at y = 0 for reference
  abline(h = 0, lty = 1)

  # Retrieve current y-axis ticks
  y_ticks <- axTicks(2)

  # Add fine dashed horizontal lines exactly at y-axis ticks
  abline(h = y_ticks, col = "gray", lty = 3, lwd = 0.5)
  abline(v = x_ticks, col = "gray", lty = 3, lwd = 0.5)

  # Title
  title(main = "")

  # Close the graphics device if file_path is provided
  if (!is.null(file_path)) {
    dev.off()
  }
}

######################
# Outcome: #
######################

# sub <- subset(dat_female, age >= 40 & age <= 70)
# sub <- sub %>% select(pid, syear, current_part_time, care_ever, time_to_treat, age)
# sub <- sub %>% arrange(pid, syear)

# selected_pids <- sub %>%
#  filter(age >= 50 + 2 & age <= 60 -2 & time_to_treat == 0) %>%
#  distinct(pid)

# dat_est <- sub %>%
#  filter(pid %in% selected_pids$pid)

# ep013_

table(dat_female$full_time, dat_female$binned_distance_to_treat)
table(dat_male$full_time, dat_male$binned_distance_to_treat)


mod_twfe_ft_women = feols(full_time ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                            mergeid + age + int_year,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          #weights = ~design_weight,
                          data = dat_female
)
iplot(mod_twfe_ft_women,
      #xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      #ylab = 'Arbeit in Vollzeit',
      xlab="Time to event (in years): Parent in bad health",
      ylab="Probability of working full-time",
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_women)

plot_event_study(mod_twfe_ft_women,
                 xlim = c(-6, 4),
                 ylim = c(-0.35, 0.25),
                 #xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 #ylab = "Veränderung der Vollzeitarbeit (in %)",
                 xlab="Time to event (in years): Parent in bad health",
                 ylab="Probability of working full-time",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
                 )


# Men
mod_twfe_ft_men = feols(full_time ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                             ## FEs
                          mergeid + age + int_year,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        #weights = ~design_weight,
                        data = dat_male
)
iplot(mod_twfe_ft_men,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Arbeit in Vollzeit',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_ft_men)

plot_event_study(mod_twfe_ft_men,
                 xlim = c(-6, 4),
                 ylim = c(-0.35, 0.25),
                 #xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 #ylab = "Veränderung der Vollzeitarbeit (in %)",
                 xlab="Time to event (in years): Parent in bad health",
                 ylab="Probability of working full-time",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)



# Hours worked
mod_twfe_hours_women = feols(ep013_ ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                            mergeid + age + int_year,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          #weights = ~design_weight,
                          data = dat_female
)
iplot(mod_twfe_hours_women,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Arbeit in Vollzeit',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_hours_women)

plot_event_study(mod_twfe_hours_women,
                 xlim = c(-6, 4),
                 ylim = c(-10, 6),
                 #xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 #ylab = "Veränderung der Arbeitsstunden",
                 xlab="Time to event (in years): Parent in bad health",
                 ylab="Working hours (weekly)",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)


mod_twfe_hours_men = feols(ep013_ ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                             ## FEs
                          mergeid + age + int_year,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        #weights = ~design_weight,
                        data = dat_male
)
iplot(mod_twfe_hours_men,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Arbeit in Vollzeit',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_hours_men)


plot_event_study(mod_twfe_hours_men,
                 xlim = c(-6, 4),
                 ylim = c(-10, 6),
                 #xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 #ylab = "Veränderung der Arbeitsstunden",
                 xlab="Time to event (in years): Parent in bad health",
                 ylab="Working hours (weekly)",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)
