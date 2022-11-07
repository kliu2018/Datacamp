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
