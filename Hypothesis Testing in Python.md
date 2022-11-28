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
## Using ttest()
~~~
# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'], y=0, alternative='two-sided')


                              
# Print the test results
print(test_results)

# Conduct a t-test on diff
test_results = pingouin.ttest(x=sample_dem_data['diff'], 
                              y=0, 
                              alternative="two-sided")

# Conduct a paired t-test on dem_percent_12 and dem_percent_16
paired_test_results = pingouin.ttest(x=sample_dem_data['dem_percent_12'], y=sample_dem_data['dem_percent_16'], paired=True, alternative='two-sided')



                              
# Print the paired test results
print(paired_test_results)
~~~
## Visualizing many categories
~~~
# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby('shipment_mode')['pack_price'].mean()

# Print the grouped means
print(xbar_pack_by_mode)

# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].mean()

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby('shipment_mode')['pack_price'].std()

# Print the grouped standard deviations
print(s_pack_by_mode)

# Calculate the mean pack_price for each shipment_mode
xbar_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].mean()

# Calculate the standard deviation of the pack_price for each shipment_mode
s_pack_by_mode = late_shipments.groupby("shipment_mode")['pack_price'].std()

# Boxplot of shipment_mode vs. pack_price
sns.boxplot(data=late_shipments, x='pack_price', y='shipment_mode')
plt.show()
~~~
## Conducting an ANOVA test
~~~
# Run an ANOVA for pack_price across shipment_mode
anova_results = pingouin.anova(data=late_shipments, dv='pack_price', between='shipment_mode')



# Print anova_results
print(anova_results)
~~~
## Pairwise t-tests
~~~
# Perform a pairwise t-test on pack price, grouped by shipment mode
pairwise_results = pingouin.pairwise_tests(data=late_shipments, dv='pack_price', between='shipment_mode', padjust='none')
# Print pairwise_results
print(pairwise_results)

# Modify the pairwise t-tests to use Bonferroni p-value adjustment
pairwise_results = pingouin.pairwise_tests(data=late_shipments, 
                                           dv="pack_price",
                                           between="shipment_mode",
                                           padjust="bonf")

# Print pairwise_results
print(pairwise_results)
~~~
## Test for single proportions
~~~
# Hypothesize that the proportion of late shipments is 6%
p_0 = 0.06

# Calculate the sample proportion of late shipments
p_hat = (late_shipments['late']=='Yes').mean()

# Calculate the sample size
n = len(late_shipments)

# Print p_hat and n
print(p_hat, n)

# Calculate the numerator and denominator of the test statistic
numerator = p_hat - p_0
denominator = np.sqrt(p_0*(1-p_0)/n)

# Calculate the test statistic
z_score = numerator / denominator

# Print the result
print(z_score)

# Calculate the p-value from the z-score
p_value = 1 - norm.cdf(z_score)
# Print the p-value
print(p_value)
~~~
## Test of two proportions
~~~
# Calculate the pooled estimate of the population proportion
p_hat = (ns['expensive'] * p_hats['expensive'] + ns['reasonable'] * p_hats['reasonable'])/(ns['expensive'] + ns['reasonable'])
# Print the result
print(p_hat)

# Calculate p_hat one minus p_hat
p_hat_times_not_p_hat = (1- p_hat) * p_hat

# Divide this by each of the sample sizes and then sum
p_hat_times_not_p_hat_over_ns = p_hat_times_not_p_hat/ns['expensive'] + p_hat_times_not_p_hat/ns['reasonable']

# Calculate the standard error
std_error = np.sqrt(p_hat_times_not_p_hat_over_ns)

# Print the result
print(std_error)

# Calculate the z-score
z_score = (p_hats['expensive'] - p_hats['reasonable'])/std_error

# Print z_score
print(z_score)

# Calculate the p-value from the z-score
p_value = 1-norm.cdf(z_score)

# Print p_value
print(p_value)
~~~
## proportions_ztest() for two samples
~~~
# Count the late column values for each freight_cost_group
late_by_freight_cost_group = late_shipments.groupby('freight_cost_group')['late'].value_counts()

# Print the counts
print(late_by_freight_cost_group)

# Create an array of the "Yes" counts for each freight_cost_group
success_counts = np.array([45, 16])

# Create an array of the total number of rows in each freight_cost_group
n = np.array([500 + 45, 439 + 16])

# Run a z-test on the two proportions
stat, p_value = proportions_ztest(success_counts, n, alternative='larger')


# Print the results
print(stat, p_value)
~~~
