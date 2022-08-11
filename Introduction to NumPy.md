# Understanding NumPy Arrays

## Your first NumPy array
~~~
# Import NumPy
import numpy as np

# Convert sudoku_list into an array
sudoku_array = np.array(sudoku_list)

# Print the type of sudoku_array 
print(type(sudoku_array))
~~~
## Creating arrays from scratch
~~~
# Create an array of zeros which has four columns and two rows
zero_array = np.zeros((2, 4))
print(zero_array)

# Create an array of random floats which has six columns and three rows
random_array = np.random.random((3, 6))
print(random_array)
~~~
## A range array
~~~
# Create an array of integers from one to ten
one_to_ten = np.arange(1,11)

# Create your scatterplot
plt.scatter(x=one_to_ten, y=doubling_array)
plt.show()
~~~
## 3D array creation
~~~
# Create the game_and_solution 3D array
game_and_solution = np.array([sudoku_game, sudoku_solution])

# Print game_and_solution
print(game_and_solution) 
~~~
## The fourth dimension
~~~
# Create a second 3D array of another game and its solution 
new_game_and_solution = np.array([new_sudoku_game, new_sudoku_solution])

# Create a 4D array of both game and solution 3D arrays
games_and_solutions = np.array([game_and_solution, new_game_and_solution])

# Print the shape of your 4D array
print(games_and_solutions.shape)
~~~
## Flattening and reshaping
~~~
# Flatten sudoku_game
flattened_game = sudoku_game.flatten()

# Print the shape of flattened_game
print(flattened_game.shape)

# Reshape flattened_game back to a nine by nine array
reshaped_game = flattened_game.reshape(9, 9)

# Print sudoku_game and reshaped_game
print(sudoku_game)
print(reshaped_game)
~~~
## The dtype argument
~~~
# Create an array of zeros with three rows and two columns
zero_array = np.zeros((3, 2))

# Print the data type of zero_array
print(zero_array.dtype)

# Create a new array of int32 zeros with three rows and two columns
zero_int_array = np.zeros((3, 2), dtype=np.int32)

# Print the data type of zero_int_array
print(zero_int_array.dtype)
~~~
## A smaller sudoku game
~~~
# Print the data type of sudoku_game
print(sudoku_game.dtype)

# Print the data type of sudoku_game
print(sudoku_game.dtype)

# Change the data type of sudoku_game to int8
small_sudoku_game = sudoku_game.astype(np.int8)

# Print the data type of small_sudoku_game
print(small_sudoku_game.dtype)
~~~
# Selecting and Updating Data
## Slicing and indexing trees
~~~
# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]
# Print the first five block_ids
print(block_ids[:5])
# Select the tenth block ID from block_ids
tenth_block_id = block_ids[9]
print(tenth_block_id)
# Select five block IDs from block_ids starting with the tenth ID
block_id_slice = block_ids[9:14]
print(block_id_slice)
~~~
## Stepping into 2D
~~~
# Create an array of the first 100 trunk diameters from tree_census
hundred_diameters = tree_census[:100, 2]
print(hundred_diameters)
# Create an array of trunk diameters with even row indices from 50 to 100 inclusive
every_other_diameter = tree_census[50:101:2, 2]
print(every_other_diameter)
~~~
## Sorting trees
~~~
# Extract trunk diameters information and sort from smallest to largest
sorted_trunk_diameters = np.sort(tree_census[:,2], axis=0)
print(sorted_trunk_diameters)
~~~
## Filtering with masks
~~~
# Create an array which contains row data on the largest tree in tree_census
largest_tree_data = tree_census[tree_census[:, 2]==51]
print(largest_tree_data)

# Slice largest_tree_data to get only the block id
largest_tree_block_id = largest_tree_data[:,1]
print(largest_tree_block_id)

# Create an array which contains row data on all trees with largest_tree_block_id
trees_on_largest_tree_block = tree_census[tree_census[:,1]==501882]
print(trees_on_largest_tree_block)
~~~

## Fancy indexing vs. np.where()
~~~
# Create the block_313879 array containing trees on block 313879
block_313879 = tree_census[tree_census[:, 1]==313879]
print(block_313879)

# Create an array which only contains data for trees on block 313879
block_313879 = tree_census[tree_census[:, 1]==313879]
print(block_313879)
~~~

## Creating arrays from conditions
~~~
# Create and print a 1D array of tree and stump diameters
trunk_stump_diameters = np.where(tree_census[:, 2]==0, tree_census[:,3], tree_census[:,2])
print(trunk_stump_diameters)
~~~

## Adding rows
~~~
# Print the shapes of tree_census and new_trees
print(tree_census.shape, new_trees.shape)
# Add rows to tree_census which contain data for the new trees
updated_tree_census = np.concatenate((tree_census, new_trees), axis=0)
print(updated_tree_census)
~~~

## Adding columns
~~~
# Print the shapes of tree_census and trunk_stump_diameters
print(tree_census.shape, trunk_stump_diameters.shape)

# Reshape trunk_stump_diameters
reshaped_diameters = trunk_stump_diameters.reshape((1000, 1))

# Concatenate reshaped_diameters to tree_census as the last column
concatenated_tree_census = np.concatenate((tree_census, reshaped_diameters), axis=1)
print(concatenated_tree_census)
~~~

## Deleting with np.delete()
~~~
# Delete the stump diameter column from tree_census
tree_census_no_stumps = np.delete(tree_census, 3, axis=1)

# Save the indices of the trees on block 313879
private_block_indices = np.where(tree_census[:, 1]==313879)

# Delete the rows for trees on block 313879 from tree_census_no_stumps
tree_census_clean = np.delete(tree_census_no_stumps, private_block_indices, axis=0)

# Print the shape of tree_census_clean
print(tree_census_clean.shape)
~~~

# Array Mathematics!
