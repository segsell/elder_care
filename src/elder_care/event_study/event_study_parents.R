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
library(coefplot)
library(ggfixest)

BAD <- 1
FEMALE <- 2


# Load data
data <- read.csv('/home/sebastian/Projects/elder_care/bld/event_study/sandbox_parents.csv')
#data <- subset(data, distance_to_treat >= -6 & distance_to_treat <= 12)

table(data$int_year)


table(data$informal_care_child)
# 0    1
# 3609  921

table(data$informal_care_daughter)



dat_raw <- as.data.table(data, TRUE)
is.data.table(dat_raw)

table(dat_raw$distance_to_treat)


# Step 2: Drop rows where distance_to_treat is -1, -3, -5, 1, 3, 5
dat_raw <- dat_raw %>%
  filter(!binned_distance_to_treat %in% c(-1, -3, -5, -7, -9, 1, 3, 5, 7, 9))

mergeids_to_drop <- dat_raw %>%
  filter(distance_to_treat == 0 & age < 75) %>%
  select(mergeid) %>%
  distinct()

dat <- dat_raw %>%
  anti_join(mergeids_to_drop, by = "mergeid")


table(dat$informal_care_daughter)


mergeids_to_drop_2 <- dat %>%
  filter(treat_ever == 0 & health == BAD) %>%
  select(mergeid) %>%
  distinct()

dat <- dat %>%
  anti_join(mergeids_to_drop_2, by = "mergeid")


table(dat$informal_care_daughter)
table(dat$informal_care_son)

table(dat$informal_care_daughter_and_other_child)
table(dat$informal_care_son_and_other_child)

#dat <- dat %>%
#  group_by(mergeid) %>%
#  mutate(distance_to_treat = if_else(treat_ever == 0, NA_real_, distance_to_treat)) %>%
#  ungroup()
dat <- dat %>%
  group_by(mergeid) %>%
  mutate(year_treated = int_year[distance_to_treat == 0][1]) %>%
  ungroup()

table(dat$year_treated, dat$distance_to_treat)
table(dat$int_year)

x_min_value <- -10
x_max_value <- 10

y_min_value <- -0.2
y_max_value <- 700

dat_all <- subset(dat, age >= 68)
dat_mother <- subset(dat, age >= 68 & gender == FEMALE)


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



# Control group: Parent never in bad health
mod_twfe_all = feols(informal_care_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by child (daughter)',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~hh_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by child (daughter)',
      main = '',
      xaxt='n'
      )
axis(1,at=c(10,30,40))
summary(mod_twfe_mother)


############
# Daughter #
############
table(dat_all$informal_care_daughter, dat_all$binned_distance_to_treat)
table(dat_mother$informal_care_son, dat_mother$binned_distance_to_treat)



mod_twfe_all = feols(informal_care_daughter ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                          |
                            #age ,                             ## FEs
                           mergeid + age,                         ## FEs
                          cluster = ~mergeid,   ## Clustered SEs
                          weights = ~hh_weight,
                          data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Tochter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)

plot_event_study(mod_twfe_all,
                 xlim = c(-6, 4),
                 ylim = c(-0.35, 0.5),
                 xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 ylab = "Veränderung der Informellen Pflege (in %)",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)




mod_twfe_mother = feols(informal_care_daughter ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Tochter',
      main = '',
      x.lim = c(-100, 100),
      y.lim = c(-100, 100))
summary(mod_twfe_mother)

plot_event_study(mod_twfe_mother,
                 xlim = c(-6, 4),
                 ylim = c(-0.5, 0.7),
                 xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 ylab = "Veränderung der Informellen Pflege (in %)",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)


#######
# Son #
#######
table(dat_all$informal_care_son, dat_all$binned_distance_to_treat)


dat_all_son <- dat_all %>%
  filter(!binned_distance_to_treat %in% c(-8, 6))

mod_twfe_all = feols(informal_care_son ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all_son
)
iplot(mod_twfe_all,
      xlab = 'Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Sohn',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)

plot_event_study(mod_twfe_all,
                 xlim = c(-6, 4),
                 ylim = c(-0.35, 0.5),
                 xlab = "Zeit zum Ereignis (in Jahren): Eltern in schlechtem Gesundheitszustand",
                 ylab = "Veränderung der Informellen Pflege (in %)",
                 #file_path="/home/sebastian/Projects/elder_care/bld/event_study/Plots/ffull_time_women.png"
)



# too few obs
table(dat_mother$informal_care_son, dat_mother$binned_distance_to_treat)
dat_mother_son <- dat_mother %>%
  filter(!binned_distance_to_treat %in% c(-8, 6))

mod_twfe_mother = feols(informal_care_son ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother_son
)
iplot(mod_twfe_mother,
      xlab = 'Zeit zum Ereignis (in Jahren): Mutter in schlechtem Gesundheitszustand',
      ylab = 'Informelle Pflege durch Sohn',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)


####################################
# With other child (unconditional) #
####################################


mod_twfe_all = feols(informal_care_daughter_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_daughter_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)


mod_twfe_all = feols(informal_care_son_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                     |
                       #age ,                             ## FEs
                       mergeid + age,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by son',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)


mod_twfe_mother = feols(informal_care_son_and_other_child ~ i(binned_distance_to_treat, treat_ever, ref = -2)
                        |
                          #age ,                              ## FEs
                          mergeid + age,                         ## FEs
                        cluster = ~mergeid,   ## Clustered SEs
                        weights = ~design_weight,
                        data = dat_mother
)
iplot(mod_twfe_mother,
      xlab = 'time to event (mother in bad health)',
      ylab = 'informal care by son',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_mother)

#########
# Sunab #
#########

# DOES NOT WORK HERE WITH DIFFERENTIAL YEAR BINNING

df = fread("https://raw.githubusercontent.com/LOST-STATS/LOST-STATS.github.io/master/Model_Estimation/Data/Event_Study_DiD/bacon_example.csv")

dat_all$yyear_treated <- copy(dat_all$year_treated)
setDT(dat_all)
dat_all[, year_treated := ifelse(treat_ever==0, 10000, yyear_treated)]
#dat_mother[, binned_distance_to_treat := ifelse(treat_ever==0, 10000, binned_distance_to_treat)]

mod_twfe_all = feols(informal_care_daughter ~ sunab(year_treated, int_year)
                     |
                       mergeid,                         ## FEs
                     cluster = ~mergeid,   ## Clustered SEs
                     weights = ~hh_weight,
                     data = dat_all
)
iplot(mod_twfe_all,
      xlab = 'time to event (mother or father in bad health)',
      ylab = 'informal care by daughter',
      main = '',
      x.lim = c(x_min_value, x_max_value),
      y.lim = c(-10000, 100000))
summary(mod_twfe_all)
