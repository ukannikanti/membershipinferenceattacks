from Evaluate import run
import statistics
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Diabetes
epsilon = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
training_samples = [100]
results = []
datasets = ["diabetes"]
target_models = ["NN", "RF", "LR"]

for eps in epsilon:
    for training_sample in training_samples:
        for dataset in datasets:
            for model in target_models:
                precisions = []
                recalls = []
                f1_scores = []
                dict = {}
                for i in range(1, 3):
                    res = run(dataset, model, eps, training_sample)
                    precisions.append(res[0])
                    recalls.append(res[1])
                    f1_scores.append(res[2])
                dict['model'] = model
                dict['epsilon'] = eps
                dict['precision'] = statistics.mean(precisions)
                dict['recall'] = statistics.mean(recalls)
                results.append(dict)

# Save the results
df = pd.DataFrame(results)
df = df.set_index(['model', 'epsilon'])

# df.to_csv("results.csv", index=False)

# Plot the results
ax = df.plot.bar(rot=0)
plt.show()