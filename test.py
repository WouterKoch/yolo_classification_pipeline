from ultralytics import YOLO
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


options = ["snow track", "mix", "other track"]
option_labels = [label.replace(" track", "") for label in options]
runs = glob.glob("runs/classify/train*/")

for run in runs:
    with open(os.path.join(run, "args.yaml")) as f:
        metadata = f.read()
        f.close()

    lines = metadata.split("\n")
    data_source = lines[3].split("(")[1].split(" train)")[0]
    data_path = metadata.split("\n")[3].split("data: ")[1]

    for testlabel in options:
        if os.path.exists(os.path.join(run, "test results for " + testlabel + ".csv")):
            continue

        model = YOLO(os.path.join(run, "weights/best.pt"))  # initialize with best.pt

        selectionset = "all"
        if testlabel == data_source or data_source == "mix":
            selectionset = "test"

        dataset_config = f"test_sets/{selectionset} {testlabel}.yaml"

        metrics = model.val(data=dataset_config, split="test")

        # save the results to a csv file
        with open(
            os.path.join(run, "test results for " + testlabel + ".csv"), "w"
        ) as f:
            f.write("metric,value\n")
            f.write(f"top1,{metrics.results_dict.get('metrics/accuracy_top1')}\n")
            f.write(f"top5,{metrics.results_dict.get('metrics/accuracy_top5')}\n")
            f.write(f"dir,{str(metrics.save_dir)}\n")
            f.close()

        # rename the directory to include the test label
        os.rename(
            str(metrics.save_dir),
            str(metrics.save_dir)
            + " "
            + str(int(time.time()))
            + " (train "
            + data_source
            + ", test "
            + testlabel
            + ")",
        )

# --------------------------------------------
# Make the accuracy plot

plotdata = []

for run in runs:
    # read the results metadata yaml as a text file
    with open(f"{run}/args.yaml") as f:
        metadata = f.read()
        f.close()

    # get the 4th line of the metadata
    lines = metadata.split("\n")
    if len(lines) < 9:
        continue

    data_source = lines[3].split("(")[1].split(" train)")[0]
    yolomodel = lines[2].split("model: yolo")[1].split("-cls.pt")[0]
    imgsize = lines[8].split("imgsz: ")[1]
    run_number = run.split("/")[-1]

    run_csv = pd.read_csv(f"{run}/results.csv", sep=",")

    datarow = {
        "run": run_number,
        "data_sources": data_source,
        "yolomodel": yolomodel,
        "imgsize": imgsize,
        "top1_accuracy": run_csv[run_csv.columns[3]].values,
        "top5_accuracy": run_csv[run_csv.columns[4]].values,
    }

    plotdata.append(datarow)


# plot a graph of the top1_accuracies over time

# prepare a plot with two x-axes, one for the training data (snow, mix, other) and one for the validation data (snow, mix, other)

fig, ax = plt.subplots(layout="constrained", figsize=(7, 4))

ax.set_xlabel("Epoch")
ax.set_ylabel("Top 1 accuracy")

legend = []
colors = {
    "snow": "blue",
    "mix": "green",
    "other": "red",
}
legend_labels = []

for plotdata_row in plotdata:
    label = plotdata_row["data_sources"].split(" - ")[0].replace(" track", "")
    ax.plot(plotdata_row["top1_accuracy"], color=colors[label], linewidth=.75)

    if label not in legend_labels:
        legend_labels.append(label) 
        # add a legend with the data sources
        legend.append(label)

ax.legend(legend)


fig.savefig(f"{run}/../top1_accuracy.png")


# --------------------------------------------
# Make the TEST accuracy plot

plotdata = []


for run in runs:
    # read the results metadata yaml as a text file
    with open(f"{run}/args.yaml") as f:
        metadata = f.read()
        f.close()

    # get the 4th line of the metadata
    lines = metadata.split("\n")
    if len(lines) < 9:
        continue

    data_source = lines[3].split("(")[1].split(" train)")[0]
    yolomodel = lines[2].split("model: yolo")[1].split("-cls.pt")[0]
    imgsize = lines[8].split("imgsz: ")[1]
    run_number = run.split("/")[-1]

    run_csv = pd.read_csv(f"{run}/results.csv", sep=",")

    for testlabel in options:
        if os.path.exists(os.path.join(run, "test results for " + testlabel + ".csv")):
            test_results = pd.read_csv(
                os.path.join(run, "test results for " + testlabel + ".csv"), sep=","
            )
                  
            datarow = {
                "run": run_number,
                "train": data_source,
                "test": testlabel,
                "yolomodel": yolomodel,
                "imgsize": imgsize,
                "top1_accuracy": run_csv[run_csv.columns[3]].values,
                "top5_accuracy": run_csv[run_csv.columns[4]].values,
                "top1_accuracy_test": float(test_results[test_results["metric"] == "top1"]["value"].values[0]),
                "top5_accuracy_test": float(test_results[test_results["metric"] == "top5"]["value"].values[0]),
            }

            plotdata.append(datarow)


# prepare a plot with two x-axes, one for the training data (snow, mix, other) and one for the validation data (snow, mix, other)
fig, ax = plt.subplots(layout="constrained", figsize=(7, 4))


ax.set_xticks(list(range(len(options) ** 2)), labels=option_labels * len(options))
ax.tick_params("x", length=0)

ax.axhline(1 / 7, linestyle="--", color="black", linewidth=0.5)

# label the classes:
sec = ax.secondary_xaxis(location=0)

triple_option_labels = option_labels * len(options)


sec.set_xticks(
    [-1] + list(range(len(options) ** 2)),
    labels=["TRAIN\n\nTEST"]
    + np.concatenate(
        [(["\n\n" + i] * len(option_labels)) for i in option_labels], axis=0
    ).tolist(),
)
sec.tick_params("x", length=0)

# lines between the classes:
sec2 = ax.secondary_xaxis(location=0)
sec2.set_xticks([-0.5, 2.5, 5.5], labels=[])
sec2.tick_params("x", length=40, width=1.5)
ax.set_xlim(-1.5, 8.5)



for plotdata_row in plotdata:
    x = options.index(plotdata_row["train"]) + options.index(plotdata_row["test"]) * 3
    ax.plot(x, plotdata_row["top1_accuracy_test"], marker="o", markersize=5)


fig.savefig(f"{run}/../top1_test_accuracy.png")
