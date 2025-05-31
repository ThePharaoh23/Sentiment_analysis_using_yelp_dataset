from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, lower, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import os
import shutil

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .getOrCreate()

# Optimize Spark parallelism settings
spark.conf.set("spark.sql.shuffle.partitions", "200")

# Load Yelp Dataset
data_path = os.path.expanduser("~/sentiment_analysis_project/yelp_academic_dataset_review.json")
df = spark.read.json(data_path)

# Preprocess Text Data
df = df.select("text", "stars")
df = df.withColumn("text", lower(col("text")))
df = df.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))

# Tokenize Text
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

# Remove Stop Words
stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
processed_df = stopwords_remover.transform(tokenized_df)

# Assign Sentiment Labels
processed_df = processed_df.withColumn(
    "label",
    when(col("stars") >= 4, 1).when(col("stars") == 3, 2).otherwise(0)
)

# Sample Data for Testing
sampled_data = processed_df.sample(fraction=0.05, seed=42).repartition(10)
sampled_data = sampled_data.filter(col("filtered_words").isNotNull() & (col("filtered_words").getItem(0).isNotNull()))

# Feature Extraction using TF-IDF
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
tf_data = hashing_tf.transform(sampled_data)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(tf_data)
rescaled_data = idf_model.transform(tf_data).select("features", "label")

# MapReduce Function for Sentiment Counts
def mapreduce_function(data):
    mapped_data = data.rdd.map(
        lambda row: ("positive" if row.label == 1 else "neutral" if row.label == 2 else "negative", 1)
    )
    reduced_data = mapped_data.reduceByKey(lambda a, b: a + b)
    return reduced_data.collect()

sentiment_counts = mapreduce_function(rescaled_data)
print("Sentiment Counts:", sentiment_counts)

# Train-Test Split
train_data, test_data = rescaled_data.randomSplit([0.8, 0.2], seed=42)

# Train Models
lr = LogisticRegression(featuresCol="features", labelCol="label")
logistic_model = lr.fit(train_data)
logistic_predictions = logistic_model.transform(test_data)
logistic_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(logistic_predictions)

nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
naive_bayes_model = nb.fit(train_data)
nb_predictions = naive_bayes_model.transform(test_data)
nb_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(nb_predictions)

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20)
random_forest_model = rf.fit(train_data)
rf_predictions = random_forest_model.transform(test_data)
rf_accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(rf_predictions)

# Create Directories for Models and Metrics
models_dir = os.path.abspath("models")
metrics_dir = os.path.abspath("metrics")

if os.path.exists(models_dir):
    shutil.rmtree(models_dir)
os.makedirs(models_dir, exist_ok=True)

if os.path.exists(metrics_dir):
    shutil.rmtree(metrics_dir)
os.makedirs(metrics_dir, exist_ok=True)

# Save Models
print("Saving Logistic Regression model...")
logistic_model.write().overwrite().save(os.path.join(models_dir, "logistic_regression_model"))
print("Saving Naive Bayes model...")
naive_bayes_model.write().overwrite().save(os.path.join(models_dir, "naive_bayes_model"))
print("Saving Random Forest model...")
random_forest_model.write().overwrite().save(os.path.join(models_dir, "random_forest_model"))

# Save Metrics
with open(os.path.join(metrics_dir, "accuracy.txt"), "w") as metrics_file:
    metrics_file.write(f"Logistic Regression Accuracy: {logistic_accuracy}\n")
    metrics_file.write(f"Naive Bayes Accuracy: {nb_accuracy}\n")
    metrics_file.write(f"Random Forest Accuracy: {rf_accuracy}\n")

# Save Precision, Recall, and F1-Score for Each Model
logistic_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(logistic_predictions)
logistic_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(logistic_predictions)
logistic_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(logistic_predictions)

nb_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(nb_predictions)
nb_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(nb_predictions)
nb_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(nb_predictions)

rf_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision").evaluate(rf_predictions)
rf_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall").evaluate(rf_predictions)
rf_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(rf_predictions)

with open(os.path.join(metrics_dir, "accuracy.txt"), "a") as metrics_file:
    metrics_file.write(f"Logistic Regression Precision: {logistic_precision}\n")
    metrics_file.write(f"Logistic Regression Recall: {logistic_recall}\n")
    metrics_file.write(f"Logistic Regression F1-Score: {logistic_f1}\n")
    metrics_file.write(f"Naive Bayes Precision: {nb_precision}\n")
    metrics_file.write(f"Naive Bayes Recall: {nb_recall}\n")
    metrics_file.write(f"Naive Bayes F1-Score: {nb_f1}\n")
    metrics_file.write(f"Random Forest Precision: {rf_precision}\n")
    metrics_file.write(f"Random Forest Recall: {rf_recall}\n")
    metrics_file.write(f"Random Forest F1-Score: {rf_f1}\n")

# Save Model Comparison as a Bar Chart
models = ["Logistic Regression", "Naive Bayes", "Random Forest"]
accuracies = [logistic_accuracy, nb_accuracy, rf_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.ylim(0, 1)
plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.savefig(os.path.join(metrics_dir, "model_comparison.png"))
plt.close()

# Save Sentiment Distribution as a Pie Chart
labels = ["Negative", "Neutral", "Positive"]
counts = [count[1] for count in sentiment_counts]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'yellow', 'green'])
plt.title("Sentiment Distribution")
plt.savefig(os.path.join(metrics_dir, "sentiment_distribution.png"))
plt.close()

# Stop Spark Session
spark.stop()