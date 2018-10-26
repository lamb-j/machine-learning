# Read data from JSON object and get relevant features
import json
import glob
import numpy as np
from pprint import pprint

match_data_files = []

for f in glob.glob("data/*.json"):
  with open(f, "rb") as infile:
    match_data_files.append(json.load(infile))


def extract_features(data_files):
  
  feature_array = []
  feature_label = []
  
  winner_c_total = 0
  winner_d_total = 0
  winner_g_total = 0

  winner_og_total = 0

  loser_c_total = 0
  loser_d_total = 0
  loser_g_total = 0

  loser_og_total = 0

  num_winners = 0
  winner_has_more_gold = 0
  winner_has_more_gold_10 = 0
  winner_has_more_tk = 0

  winner_has_more_gpm10 = 0

  num_A_winners = 0
  num_B_winners = 0

  # iterate over each data file
  for data in data_files:
  
    # iterate over each match 
    for match_id in range(0, len(data["matches"]) ):
      match_data = data["matches"][match_id]
    
      # creeps per minute
      A_cpm = []
      B_cpm = []

      # damage taken per minute
      A_dpm = []
      B_dpm = []

      # gold per minute
      A_gpm = []
      B_gpm = []

      # total gold
      A_og = 0
      B_og = 0

      # tower kills
      A_tk = 0 
      B_tk = 0

      # gold per minute, different calculation
      A_gpm10 = 0
      B_gpm10 = 0
    
      for i in range(0, 5):
        # Get individual creeps per minute (cpm)
        player_timeline = match_data["participants"][i]["timeline"]

        A_cpm.append ( player_timeline["creepsPerMinDeltas"]["zeroToTen"] )
        A_dpm.append ( player_timeline["damageTakenPerMinDeltas"]["zeroToTen"] )
        A_gpm.append ( player_timeline["goldPerMinDeltas"]["zeroToTen"] )

        A_gpm10 += match_data["participants"][i]["timeline"]["goldPerMinDeltas"]["zeroToTen"] 
        A_og += match_data["participants"][i]["stats"]["goldEarned"]
        A_tk += match_data["participants"][i]["stats"]["towerKills"]
        
      
      for i in range(5, 10):
        player_timeline = match_data["participants"][i]["timeline"]

        # Get individual creeps per minute (cpm)
        B_cpm.append ( player_timeline["creepsPerMinDeltas"]["zeroToTen"] )
        B_dpm.append ( player_timeline["damageTakenPerMinDeltas"]["zeroToTen"] )
        B_gpm.append ( player_timeline["goldPerMinDeltas"]["zeroToTen"] )
      
        B_gpm10 += match_data["participants"][i]["timeline"]["goldPerMinDeltas"]["zeroToTen"] 
        B_og += match_data["participants"][i]["stats"]["goldEarned"]
        B_tk += match_data["participants"][i]["stats"]["towerKills"]
      
      # generate features from team A data
      A_cpm_total = sum(A_cpm)
      A_cpm_min = min(A_cpm)
      A_cpm_max = max(A_cpm)
      A_cpm_avg = np.mean(A_cpm) 
      A_cpm_var = np.var(A_cpm) 

      A_dpm_total = sum(A_dpm)
      A_dpm_min = min(A_dpm)
      A_dpm_max = max(A_dpm)
      A_dpm_avg = np.mean(A_dpm) 
      A_dpm_var = np.var(A_dpm) 

      A_gpm_total = sum(A_gpm)
      A_gpm_min = min(A_gpm)
      A_gpm_max = max(A_gpm)
      A_gpm_avg = np.mean(A_gpm) 
      A_gpm_var = np.var(A_gpm) 

      A_label = match_data["participants"][0]["stats"]["winner"]
    
      # Add team A performances to feature array
      A_item = [   A_cpm_avg, A_cpm_max, A_cpm_avg, A_cpm_var,
                   A_dpm_avg, A_dpm_max, A_dpm_avg, A_dpm_var,
                   A_gpm_avg, A_gpm_max, A_gpm_avg, A_gpm_var]
      feature_array.append(A_item) 
      feature_label.append(A_label) 
    
      # generate features from team B data
      B_cpm_total = sum(B_cpm)
      B_cpm_min = min(B_cpm)
      B_cpm_max = max(B_cpm)
      B_cpm_avg = np.mean(B_cpm) 
      B_cpm_var = np.var(B_cpm) 

      B_dpm_total = sum(B_dpm)
      B_dpm_min = min(B_dpm)
      B_dpm_max = max(B_dpm)
      B_dpm_avg = np.mean(B_dpm) 
      B_dpm_var = np.var(B_dpm) 

      B_gpm_total = sum(B_gpm)
      B_gpm_min = min(B_gpm)
      B_gpm_max = max(B_gpm)
      B_gpm_avg = np.mean(B_gpm) 
      B_gpm_var = np.var(B_gpm) 

      B_label = match_data["participants"][5]["stats"]["winner"]
    
      # Add team B performances to feature array
      B_item = [   B_cpm_avg, B_cpm_max, B_cpm_avg, B_cpm_var,
                   B_dpm_avg, B_dpm_max, B_dpm_avg, B_dpm_var,
                   B_gpm_avg, B_gpm_max, B_gpm_avg, B_gpm_var]
      feature_array.append(B_item)
      feature_label.append(B_label)

#      if A_gpm_total != A_gpm10:
#        print "A_gpm_total", A_gpm_total, "A_gpm_10", A_gpm10
#      
#      if B_gpm_total != B_gpm10:
#        print "B_gpm_total", B_gpm_total, "B_gpm_10", B_gpm10
#
#      #if A won
#      if A_label:
#        winner_c_total += A_cpm_total 
#        winner_d_total += A_dpm_total 
#        winner_g_total += A_gpm_total 
#
#        winner_og_total += A_og
#
#        loser_c_total += B_cpm_total 
#        loser_d_total += B_dpm_total 
#        loser_g_total += B_gpm_total 
#
#        loser_og_total += B_og
#
#        num_winners += 1
#        if A_og > B_og:
#          winner_has_more_gold += 1
#
#        if A_tk > B_tk:
#          winner_has_more_tk += 1
#
#        if A_gpm10 > B_gpm10:
#          winner_has_more_gpm10 += 1
#
#        num_A_winners += 1
#
#      else:
#        winner_c_total += B_cpm_total 
#        winner_d_total += B_dpm_total 
#        winner_g_total += B_gpm_total 
#
#        winner_og_total += B_og
#
#        loser_c_total += A_cpm_total 
#        loser_d_total += A_dpm_total 
#        loser_g_total += A_gpm_total 
#
#        loser_og_total += A_og
#
#        num_winners += 1
#        if B_og > A_og:
#          winner_has_more_gold += 1
#
#        if B_tk > A_tk:
#          winner_has_more_tk += 1
#
#        if B_gpm10 > A_gpm10:
#          winner_has_more_gpm10 += 1
#
#        num_B_winners += 1
  
  # convert labels from true/false to 0/1
  feature_label = map(int, feature_label)

#  print "winner_c_total", winner_c_total
#  print "winner_d_total", winner_d_total
#  print "winner_g_total", winner_g_total
#  print "winner_og_total", winner_og_total
#
#  print ""
#  print "loser_c_total", loser_c_total
#  print "loser_d_total", loser_d_total
#  print "loser_g_total", loser_g_total
#  print "loser_og_total", loser_og_total
#  print ""
#  print "num_winners", num_winners
#  print "num_A_winners", num_A_winners
#  print "num_B_winners", num_B_winners
#  print "winner_has_more_gold", winner_has_more_gold
#  print "winner_has_more_tk", winner_has_more_tk
#  print "winner_has_more_gpm10", winner_has_more_gpm10

  return feature_array, feature_label

# Extract Features from Data
feature_names = [ "cpm_min", "cpm_max", "cpm_avg", "cpm_var",
                  "dpm_min", "dpm_max", "dpm_avg", "dpm_var",
                  "gpm_min", "gpm_max", "gpm_avg", "gpm_var" ]
feature_targets = ["lose", "win"]

train_data = match_data_files[0:7]

# validate 0, 1
#train_data = match_data_files[2:7]
#validate_data = match_data_files[0:2]

# validate 2, 3
#train_data = match_data_files[0:2] + match_data_files[4:7]
#validate_data = match_data_files[2:4]

# validate 4, 5
#train_data = match_data_files[0:4] + match_data_files[6:7]
#validate_data = match_data_files[4:6]

# validate 7
#train_data = match_data_files[0:6] 
#validate_data = match_data_files[6:7]

test_data = match_data_files[7 : 9]

train_array, train_label = extract_features(train_data)
#validate_array, validate_label = extract_features(validate_data)
test_array, test_label = extract_features(test_data)

print "--- Dataset ---"
print "Number of features:", len(feature_names)
print "train size:", len(train_array)
#print "validate size:", len(validate_array)
print "test size:", len(test_array)
print ""

# Decision Tree 
from sklearn import tree
tree_depth = 3
clf = tree.DecisionTreeClassifier( max_depth = tree_depth, random_state = 0)
clf = clf.fit( train_array, train_label )

# export tree
import pydotplus
dot_data = tree.export_graphviz(clf, 
    out_file = None, 
    feature_names=feature_names,
    class_names=feature_targets,
    filled=True, rounded=True,
    special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("dtree.pdf")

train_accuracy = clf.score(train_array, train_label)
#validate_accuracy = clf.score(validate_array, validate_label)
test_accuracy = clf.score(test_array, test_label)

print "--- Decision Tree Classifier ---"
print "tree_depth", tree_depth
print "train accuracy:", train_accuracy
#print "validate accuracy:", validate_accuracy
print "test accuracy:", test_accuracy
print ""

# Bagged Decision Tree 
from sklearn import ensemble 
tree_depth = 3
est = 15
clf = ensemble.BaggingClassifier (tree.DecisionTreeClassifier( max_depth = tree_depth),
                                  max_samples = 1.0,
                                  max_features = 1.0,
                                  n_estimators = est)

clf = clf.fit( train_array, train_label )

train_accuracy = clf.score(train_array, train_label)
#validate_accuracy = clf.score(validate_array, validate_label)
test_accuracy = clf.score(test_array, test_label)

print "--- Bagging Tree Classifier ---"
print "n_estimators:", est 
print "train accuracy:", train_accuracy
#print "validate accuracy:", validate_accuracy
print "test accuracy:", test_accuracy
print ""

# Random Forest 
tree_depth = 3
est = 15

clf = ensemble.RandomForestClassifier(n_estimators=est, 
    criterion='gini', 
    max_depth=tree_depth, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=4, 
    max_leaf_nodes=None, 
    min_impurity_split=1e-07, 
    bootstrap=True, 
    oob_score=False, 
    n_jobs=1, 
    random_state=None, 
    verbose=0, 
    warm_start=False, 
    class_weight=None)

clf = clf.fit( train_array, train_label )

train_accuracy = clf.score(train_array, train_label)
#validate_accuracy = clf.score(validate_array, validate_label)
test_accuracy = clf.score(test_array, test_label)

print "--- Random Forest Classifier ---"
print "n_estimators:", est
print "train accuracy:", train_accuracy
#print "validate accuracy:", validate_accuracy
print "test accuracy:", test_accuracy
print ""
