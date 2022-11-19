# Introduction to Hypothesis Testing
## Calculating the sample mean
~~~
# Print the late_shipments dataset
print(late_shipments)

# Calculate the proportion of late shipments
late_prop_samp = (late_shipments['late']=='Yes').mean()

# Print the results
print(late_prop_samp)
~~~
## Calculating a z-score

~~~
# Hypothesize that the proportion is 6%
late_prop_hyp = 0.06

# Calculate the standard error
std_error = np.std(late_shipments_boot_distn, ddof=1)

# Find z-score of late_prop_samp
z_score = (late_prop_samp.mean()-late_prop_hyp) / std_error

# Print z_score
print(z_score)
~~~
## Calculating p-values
~~~
# Calculate the z-score of late_prop_samp
z_score = (late_prop_samp - late_prop_hyp) / std_error

# Calculate the p-value
p_value = 1- norm.cdf(z_score, loc=0, scale=1)
                 
# Print the p-value
print(p_value) 
~~~
## Calculating a confidence interval
~~~
# Calculate 95% confidence interval using quantile method
lower = np.quantile(late_shipments_boot_distn, 0.025)
upper = np.quantile(late_shipments_boot_distn, 0.975)

# Print the confidence interval
print((lower, upper))
~~~
# Two-Sample and ANOVA Tests
## Two sample mean test statistic
~~~
# Calculate the numerator of the test statistic
numerator = xbar_yes - xbar_no

# Calculate the denominator of the test statistic
denominator = np.sqrt(s_yes**2/n_yes + s_no**2/n_no)

# Calculate the test statistic
t_stat = numerator/denominator

# Print the test statistic
print(t_stat)
~~~
## From t to p
~~~
# Calculate the degrees of freedom
degrees_of_freedom = n_no + n_yes -2

# Calculate the p-value from the test stat
p_value = t.cdf(t_stat, df=degrees_of_freedom)

# Print the p_value
print(p_value)
~~~
## Visualizing the difference
~~~
# Calculate the differences from 2012 to 2016
sample_dem_data['diff'] = sample_dem_data['dem_percent_12'] - sample_dem_data['dem_percent_16']

# Find the mean of the diff column
xbar_diff = sample_dem_data['diff'].mean()

# Find the standard deviation of the diff column
s_diff = sample_dem_data['diff'].std()

# Plot a histogram of diff with 20 bins
sample_dem_data['diff'].hist(bins=20)
plt.show()
~~~
