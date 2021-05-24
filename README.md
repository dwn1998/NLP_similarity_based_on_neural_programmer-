# NLP_similarity_based_on_neural_programmer-
NLP_similarity_based_on_neural_programmer  

first build two extra directory out of the code directory ../data and ../models
generate_datasets.py: create the big data set, with the set of my own, it can be runned in command with "python3 generate_datasets.py single 1000 ../data/training_set_1000.txt"

train.py: run in the command "python3 train.py" to train the neural programmer, with the change of epoch range in the code, I can control the number of epochs.
In my code I used 10 epochs.

eval.py: test the testing_data_10(same size as the training_data) and testing_data_20(different size with the training_data) based on the trained model. Calculated the accuracy of the test in trained model.

draw_epoch_5.py,draw_epoch_15.py,draw_epoch_25.py: Draw the learning curve  of the three configurations I used during training, training iteration on the x-axis, and loss on the y-axis.

similarity.py: calculate the similarity of the sentences in split_testing_data_10.txt and testing_eval_10.txt in two ways: original word frequency similarity calculation, and the similarity calculation with the neural programmer answer.
Then write the two similarity "similarity_with_eval": xxx, "similarity_of_original_dataset": xxx, in the file similarity.txt with q1 and q2.
