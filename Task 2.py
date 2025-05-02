# Databricks notebook source
# MAGIC %md
# MAGIC ##Benyamin Askari
# MAGIC ##Student ID: 00790065

# COMMAND ----------

# MAGIC %md
# MAGIC # **Task 2** - Steam 200k Dataset 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Table of Contents
# MAGIC
# MAGIC ## 1. Load Data
# MAGIC - 1.1 Header Inspection
# MAGIC - 1.2 Load Dataset with Column Names
# MAGIC
# MAGIC ## 2. Exploratory Data Analysis (EDA)
# MAGIC - 2.1 Data Summary
# MAGIC - 2.2 Missing Values
# MAGIC - 2.3 Duplicate Values
# MAGIC   - 2.3.1 Row-based Duplicates
# MAGIC   - 2.3.2 Game-level Duplicates per User
# MAGIC   - 2.3.3 Action-level Duplicates
# MAGIC   - 2.3.4 Sum of Duplicates
# MAGIC   - 2.3.5 Action Type Validation
# MAGIC - 2.4 Detecting Unusual Values
# MAGIC   - 2.4.1 Invalid User IDs
# MAGIC   - 2.4.2 Invalid Game Names
# MAGIC   - 2.4.3 Invalid Value Combinations
# MAGIC
# MAGIC ## 3. Game-User Behavior Patterns
# MAGIC - 3.1 Import and Filtering Setup
# MAGIC - 3.2 Game-Level Metrics
# MAGIC   - 3.2.1 Total Purchases per Game
# MAGIC   - 3.2.2 Total Play Instances per Game
# MAGIC   - 3.2.3 Total Play Hours per Game
# MAGIC   - 3.2.4 Purchase Frequency Segments
# MAGIC   - 3.2.5 Purchased but Not Played
# MAGIC   - 3.2.6 Purchased and Played
# MAGIC   - 3.2.7 Played but Not Purchased
# MAGIC   - 3.2.8 Average Playtime per User
# MAGIC   - 3.2.9 Maximum Playtime per User
# MAGIC   - 3.2.10 Minimum Playtime per User
# MAGIC   - 3.2.11 Final Game-Level Summary
# MAGIC - 3.3 User-Level Interaction Distribution
# MAGIC - 3.4 Sparsity
# MAGIC - 3.5 Correlation Checks
# MAGIC   - 3.5.1 Purchase vs Play Instances
# MAGIC   - 3.5.2 Purchase vs Play Hours
# MAGIC   - 3.5.3 Play Instances vs Play Hours
# MAGIC   - 3.5.4 User Purchase vs Play Hours
# MAGIC   - 3.5.5 User Purchase vs Play Instances
# MAGIC   - 3.5.6 User Play Hours vs Play Instances
# MAGIC - 3.6 Top Games and Users by Engagement Metrics
# MAGIC
# MAGIC ## 4. Recommender System – ALS
# MAGIC - 4.1 ALS on Full Dataset
# MAGIC   - 4.1.1 Indexing and Log Transformation
# MAGIC   - 4.1.2 Train-Test Split
# MAGIC   - 4.1.3 MLflow and Evaluator Setup
# MAGIC   - 4.1.4 Hyperparameter Tuning and MLflow Logging
# MAGIC   - 4.1.5 Load Best Model & Predict
# MAGIC   - 4.1.6 MAE and Residual Analysis
# MAGIC   - 4.1.7 Actual vs Predicted Plot
# MAGIC   - 4.1.8 Per-User RMSE Distribution
# MAGIC   - 4.1.9 Game Recommendation Using ALS
# MAGIC - 4.2 ALS on Filtered Dataset
# MAGIC   - 4.2.1 User Filtering
# MAGIC   - 4.2.2 Filtering Dataset and Preview
# MAGIC   - 4.2.3 Filtered Train-Test Split
# MAGIC   - 4.2.4 Create Filtered MLflow Experiment
# MAGIC   - 4.2.5 Train Filtered ALS Model with MLflow
# MAGIC   - 4.2.6 Load Best Filtered Model and Evaluate
# MAGIC   - 4.2.7 Actual vs Predicted Plot (Filtered)
# MAGIC   - 4.2.8 Per-User RMSE (Filtered)
# MAGIC   - 4.2.9 Display Recommendations
# MAGIC - 4.3 Model Comparison
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Header Inspection
# MAGIC Before loading the dataset, it is crucial to understand its structure. As shown, the dataset does not have header and we need to assign headers manually

# COMMAND ----------

# Loading the raw CSV content as plain text to inspect its structure
raw_df = spark.read.text("/FileStore/tables/steam.csv")
raw_df.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Load dataset with column names
# MAGIC Since the raw dataset lacks a header row, we manually define the column schema to correctly label the fields:
# MAGIC
# MAGIC ****.**** user_id: Unique identifier for each Steam user.
# MAGIC
# MAGIC ****.**** game: Name of the game interacted with.
# MAGIC
# MAGIC ****.**** action: Type of user interaction, either purchase or play.
# MAGIC
# MAGIC value: Encoded behaviour. 1 for purchases, and a float for hours played.

# COMMAND ----------

# Assigning column names manually since the CSV file does not include a header row
column_names = ['user_id', 'game', 'action', 'value']
steam_df = spark.read.csv("/FileStore/tables/steam.csv", header=False, inferSchema=True).toDF(*column_names)
display(steam_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploratory Data Analysis (EDA)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Data summary
# MAGIC The objective here is to capture a high-level understanding of dataset size and distribution. By computing the number of rows, distinct users, and distinct games, we evaluate the dataset for models such as ALS.

# COMMAND ----------

# Basic statistics about the dataset: total rows, columns, unique users, and unique games
total_rows = steam_df.count()
total_columns = len(steam_df.columns)
unique_users = steam_df.select("user_id").distinct().count()
unique_games = steam_df.select("game").distinct().count()

# COMMAND ----------

print("Dataset Summary:")
print(f"- Total Rows: {total_rows}")
print(f"- Total Columns: {total_columns}")
print(f"- Unique Users: {unique_users}")
print(f"- Unique Games: {unique_games}")

# COMMAND ----------

# MAGIC %md
# MAGIC **This structure confirms that we are dealing with a large dataset. there are over 12,000 users and 5,000 games, and there are  200k interactions — which could mean that not alot of user-game interactions are observed, reinforcing the need for a model that can work well under sparsity.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Missing values

# COMMAND ----------

# MAGIC %md
# MAGIC **This code snippet systematically checks for the presence of missing values (nulls) across all columns of the steam_df DataFrame using Spark SQL. it builds a DataFrame containing the null counts for each column, then renders the result visually in Databricks' notebook interface.**

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, isnan
# Counting nulls for each column in the DataFrame
null_counts = steam_df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c + "_nulls")
    for c in steam_df.columns
])
display(null_counts)

# COMMAND ----------

# MAGIC %md
# MAGIC **This confirms that every interaction is fully defined by a valid user ID, game name, action type, and value. This is especially crucial in a recommender system, where even a single null in user_id, game, or value could prevent the ALS algorithm from assigning unique integer indices or calculating ratings accurately.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Duplicate values

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.1 Row-based duplicates

# COMMAND ----------

# MAGIC %md
# MAGIC **This segment identifies exact duplicate interaction records. It uses the .groupBy(...) method on all columns; user_id, game, action, and value — followed by .count() to identify frequency. The .filter(col("count") > 1) step then isolates rows that occur more than once.**
# MAGIC
# MAGIC **This method only captures entries that are identical across all attributes and have been repeated.**

# COMMAND ----------

# Identifying exact duplicate rows based on all columns
row_level_duplicates_df = steam_df.groupBy(steam_df.columns).count().filter(col("count") > 1)
display(row_level_duplicates_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **As evident, thee are many users with duplicated purchases. this means the user have purchased the game twice**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **This second snippet is a diagnostic step to manually inspect behaviour from one of the flagged users (305835588). It helps validate that duplication is not random or corrupted but repeated purchases.**

# COMMAND ----------

# Inspecting repeated game entries for a specific user
user_305835588_df = steam_df.filter(col("user_id") == "305835588")
display(user_305835588_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.2 Game-level duplicates per user
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This step generalizes duplication detection from entire row-matching (Section 2.3.1) to user-game pairs, regardless of the action type or value. This logic helps quantify how deeply users are interacting with specific titles — a crucial metric in implicit feedback systems.**

# COMMAND ----------

# Counting how many times the same user interacted with the same game
game_dupes_per_user_df = steam_df.groupBy("user_id", "game").count().filter(col("count") > 1)
display(game_dupes_per_user_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **This query inspects a specific case from the flagged dataset to explore whether these repetitions are random or behaviourally valid.**

# COMMAND ----------

# Viewing all records for another specific user
user_118664413_df = steam_df.filter(col("user_id") == "118664413")
display(user_118664413_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Some users like (118664413) have interacted with the same game (e.g., Grand Theft Auto San Andreas) 3 or 4 times. These entries mix purchase and play actions, each logged individually.**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.3 Action-level duplicates
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This block identifies duplicate interaction types per user and game. Rather than just grouping by user_id and game, this query includes action, allowing us to isolate how many times a specific behaviour (e.g., multiple purchases or multiple play sessions) has been logged and ensures only duplicated combinations are shown. crucial for confirming behavioural frequency and repetition.**

# COMMAND ----------

# Identifying user-game-action combinations that occur more than once
action_dupes_per_user_game_df = steam_df.groupBy("user_id", "game", "action").count().filter(col("count") > 1)
display(action_dupes_per_user_game_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Repeated purchase actions dominate the duplications. Most duplicates are 2, indicating multiple purchases.**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.4 Sum of duplicates
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This step finalizes the deduplication process by merging repeated actions into meaningful metrics:**
# MAGIC
# MAGIC All purchase events for the same user-game pair are counted → a proxy for how many times a game was bought or licensed.
# MAGIC
# MAGIC All play sessions are summed → this captures total playtime, maintaining full fidelity of the user’s actual engagement.
# MAGIC
# MAGIC ****.**** Multiple purchases may reflect version upgrades or re-acquisitions across devices.
# MAGIC
# MAGIC ****.**** Multiple play sessions are the natural unit of behavioural data and should be preserved, not averaged or discarded.
# MAGIC
# MAGIC ****.**** By converting these repetitions into cumulative counts (purchase) or continuous usage (play), the model can later infer latent user interests far more accurately.

# COMMAND ----------

# MAGIC %md
# MAGIC **The use of .cast("double") ensures numeric compatibility across unioned datasets. Sorting and temporary view creation (p_steam) make the result readily reusable for SQL or visualization.**

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, count as spark_count

# If a user purchased the same game multiple times, sum up the purchase count
purchase_df = steam_df.filter(col("action") == "purchase") \
    .groupBy("user_id", "game", "action") \
    .agg(spark_count("*").cast("double").alias("value"))

# If a user played the same game multiple times, sum the total hours without rounding
play_df = steam_df.filter(col("action") == "play") \
    .groupBy("user_id", "game", "action") \
    .agg(spark_sum("value").alias("value"))

# Merge normalized purchase and play data into a clean, de-duplicated dataset
p_steam_df = purchase_df.union(play_df).orderBy("user_id", "game", "action")
p_steam_df.createOrReplaceTempView("p_steam")

# Display the cleaned and aggregated dataset
display(p_steam_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **This block performs a targeted integrity check for a user previously identified with duplicated entries. By querying the final de-duplicated DataFrame (p_steam_df), we ensure the aggregation logic applied in 2.3.4 was successful.**

# COMMAND ----------

# Confirm that duplicates have been resolved correctly
user_118664413_df = p_steam_df.filter(col("user_id") == "118664413")
display(user_118664413_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **The output now shows:**
# MAGIC
# MAGIC A single row per user-game-action combination.
# MAGIC Summed play hours, e.g., "GTA San Andreas" = 2.1 hours.
# MAGIC Counted purchases, e.g., same title = 2 purchases.
# MAGIC Each interaction now uniquely and fully describes user engagement, whether through repeated gameplay or re-acquisition events.
# MAGIC
# MAGIC ****.**** The data cleaning phase is now complete and the final dataset is free of row-level and action-level redundancy. Retains meaningful distinctions across repeat interactions.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.5
# MAGIC **Before segmenting behavioral trends or building interaction matrices, it’s important to verify the range of action types logged in the dataset. This check ensures:**
# MAGIC
# MAGIC ****.**** All expected user behaviours are present.
# MAGIC
# MAGIC ****.**** No corrupted or miscellaneous action labels exist (e.g., "played", "purchased" vs "play", "purchase").

# COMMAND ----------

# Viewing all distinct action types present
distinct_actions_df = p_steam_df.select("action").distinct()
display(distinct_actions_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Exactly two interaction types are logged: purchase and play. This matches the schema and modelling assumptions made during deduplication and upcoming matrix factorization. No additional cleaning or label mapping is required.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Detecting unusual values
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.1 Invalid user_ids
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This check is a safeguard against format inconsistencies or data corruption. While Spark inferred user_id as integers, malformed imports or schema-free ingestion (e.g., CSV logs) can still result in:**
# MAGIC
# MAGIC ****.**** Non-numeric values
# MAGIC
# MAGIC ****.**** Abnormally long IDs (e.g., stringified UUIDs)
# MAGIC
# MAGIC ****.**** Anomalies caused by padding, encoding, or legacy system merges
# MAGIC
# MAGIC Checking the string length of user_id values helps validate uniformity and ensure no outliers exist before indexing for matrix factorization.

# COMMAND ----------

from pyspark.sql.functions import length

# Finding long user_id strings
user_id_len_df = p_steam_df.select("user_id", length("user_id").alias("length")).orderBy(col("length").desc()).limit(10)
display(user_id_len_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **As evident all user IDs examined have a length of 9 digits.**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.2 Invalid game names
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This check focuses on identifying game title anomalies — specifically those with abnormally long string lengths. This is important for three reasons:**
# MAGIC
# MAGIC ****.**** Data integrity: Very long strings may indicate corrupted records (e.g., URLs or logs accidentally recorded as game titles).****.**** 
# MAGIC
# MAGIC ****.**** Performance: Extremely long categorical strings can cause unnecessary memory usage in encoding.****.**** 
# MAGIC
# MAGIC ****.**** Interpretability: Downstream visualizations, tooltips, and logs may be truncated or distorted by oversized names.****.**** 
# MAGIC
# MAGIC here we are more focused on the **Data integrity**

# COMMAND ----------

# Finding long game names
game_len_df = p_steam_df.select("game", length("game").alias("length")).orderBy(col("length").desc()).limit(10)
display(game_len_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **There are no malformed, empty, or system-level strings in the top 10 entries. The name itself, while long, is valid — matching known game entries in Steam’s catalog.**

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.3 Invalid value combinations
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **This diagnostic step ensures that interaction values adhere to their predefined value or threshold**
# MAGIC
# MAGIC ****.**** purchase values inititaly where set as 1. However since we preprocessed their data now they can store other non zero values too. in this step, we are ckecking for non 1 values to see if any inordinary values are stored for purchase.
# MAGIC
# MAGIC ****.**** play values must be positive; negative or zero values would indicate invalid tracking, possibly system or logging errors.
# MAGIC
# MAGIC Such checks are vital for preserving interpretability and learning fidelity in collaborative filtering algorithms, where value is used as the implicit feedback strength.

# COMMAND ----------

# Flaging rows where purchase values are not 1 or play values are not positive
invalid_value_df = p_steam_df.filter(
    ((col("action") == "purchase") & (col("value") != 1)) |
    ((col("action") == "play") & (col("value") <= 0))
)
display(invalid_value_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **As evident, the flagged cases all have purchase values of 2.**
# MAGIC
# MAGIC Importantly:
# MAGIC
# MAGIC ****.**** These reflect intentional design choices from earlier aggregation.
# MAGIC
# MAGIC ****.**** Users such as 2259650 appear to have purchased the same title multiple times; which we have chosen to retain as a signal of repeated interest.
# MAGIC
# MAGIC ****.**** There are no play entries with zero or negative values, another indicator of dataset cleanliness.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Game-User Behavior Patterns
# MAGIC To support meaningful recommendation modeling and behavioral analysis, this section extracts per-game statistics based on user interactions. These aggregated metrics will be used to:
# MAGIC
# MAGIC *****.***** Profile popular
# MAGIC
# MAGIC *****.***** Understand play-purchase relationships
# MAGIC
# MAGIC *****.***** Inform evaluation criteria (e.g., popularity in ALS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Import and Filtering Setup

# COMMAND ----------

# MAGIC %md
# MAGIC **Two filtered subsets of the master dataset p_steam_df are created:**
# MAGIC
# MAGIC *.* play_only_df: Contains only rows where users played a game.
# MAGIC
# MAGIC *.* purchase_only_df: Contains only rows where users purchased a game.
# MAGIC
# MAGIC This modular design improves performance and code readability for all subsequent calculations.

# COMMAND ----------

from pyspark.sql.functions import col, sum as spark_sum, count as spark_count, countDistinct, avg as spark_avg, max as spark_max, min as spark_min

# Create filtered DataFrames to reuse for all play-based and purchase-based calculations
play_only_df = p_steam_df.filter(col("action") == "play")
purchase_only_df = p_steam_df.filter(col("action") == "purchase")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Game-Level Metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.1 Total Number of Purchases per Game
# MAGIC **This block computes how many times each game was purchased in total:**
# MAGIC
# MAGIC . groupBy("game") groups rows by game title.
# MAGIC
# MAGIC . spark_sum("value") adds up the value column — which was earlier set to count repeated purchases.
# MAGIC
# MAGIC The result is stored in a new column **total_purchase**.

# COMMAND ----------

# 1. Total number of purchases per game
purchase_counts_df = purchase_only_df \
    .groupBy("game") \
    .agg(spark_sum("value").alias("total_purchase"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.2 Total Number of Play Instances per Game
# MAGIC **Here, we measure the number of play sessions logged per game, regardless of hours:**
# MAGIC
# MAGIC .spark_count("action") counts how many play actions were recorded for each title.
# MAGIC
# MAGIC .This provides a discrete frequency metric complementary to total hours played.
# MAGIC
# MAGIC This will later be paired with **average, max, and min playtime** statistics to form a complete interaction profile.

# COMMAND ----------

# 2. Total number of play instances per game
play_instance_df = play_only_df \
    .groupBy("game") \
    .agg(spark_count("action").alias("total_play_instance"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.3 Total Number of Hours Played per Game
# MAGIC **This metric captures total engagement intensity per game:**
# MAGIC
# MAGIC While play instance count shows frequency, this measures depth, how long users actually played.
# MAGIC
# MAGIC ******.****** spark_sum("value") totals the cumulative hours for each game.
# MAGIC
# MAGIC This gives a strong indication of game popularity post-purchase and is essential for correlating purchase conversion to usage.

# COMMAND ----------

# 3. Total number of hours played per game
play_hours_df = play_only_df \
    .groupBy("game") \
    .agg(spark_sum("value").alias("total_play_hours"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.4 Games Purchased Once or Twice per User
# MAGIC **First**, this computes how many times each user purchased a specific game.
# MAGIC **Then**, filters are applied to split the dataset into two meaningful segments:
# MAGIC
# MAGIC ******.****** One-time buyers — users with minimal engagement
# MAGIC
# MAGIC ******.****** Two-time buyers — indicating potential for DLC upgrades, repurchases, or gifting
# MAGIC
# MAGIC **Finaly**, the code quantifies how many unique users per game fall into the one-time or two-time buyer categories. These metrics are particularly useful for:
# MAGIC
# MAGIC ******.****** Understanding purchase churn rates
# MAGIC
# MAGIC ******.****** Differentiating casual titles from games with follow-up interest

# COMMAND ----------

# 4. Games purchased once or twice per user
game_user_purchase_counts = purchase_only_df \
    .groupBy("user_id", "game") \
    .agg(spark_sum("value").alias("total_purchases"))

# Games purchased once per user
games_purchased_once = game_user_purchase_counts.filter(col("total_purchases") == 1)

# Games purchased twice per user
games_purchased_twice = game_user_purchase_counts.filter(col("total_purchases") == 2)

# Group by game to count how many users purchased it once or twice
once_per_game_df = games_purchased_once.groupBy("game").agg(spark_count("user_id").alias("purchased_once_by_users"))
twice_per_game_df = games_purchased_twice.groupBy("game").agg(spark_count("user_id").alias("purchased_twice_by_users"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.5 Games Purchased but Not Played
# MAGIC **This query detects games that were bought but never played:**
# MAGIC
# MAGIC ******.****** The left_anti join filters purchases that do not have matching play events.
# MAGIC
# MAGIC This pattern could suggest buyer’s remorse, gifting, idling, or misaligned expectations.
# MAGIC
# MAGIC ******.****** purchase_not_played_per_game quantifies how widespread this is per game.

# COMMAND ----------

# 5. Games purchased but not played by that user
user_game_purchases = purchase_only_df.select("user_id", "game")
user_game_plays = play_only_df.select("user_id", "game")

purchased_not_played_df = user_game_purchases.join(user_game_plays, on=["user_id", "game"], how="left_anti")

# Count how many users did this per game
purchase_not_played_per_game = purchased_not_played_df.groupBy("game").agg(spark_count("user_id").alias("purchase_not_played"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.6 Games Purchased and Played
# MAGIC **This block identifies the overlap:** games that were both bought and played.
# MAGIC
# MAGIC *****.***** An inner join retains only records where the same user has both a purchase and play log for the same game.
# MAGIC
# MAGIC This is a strong confirmation of engagement, and is a valuable metric for measuring game satisfaction and alignment.

# COMMAND ----------

# 6. Games purchased and played
purchased_and_played_df = user_game_purchases.join(user_game_plays, on=["user_id", "game"], how="inner")
purchased_and_played_per_game = purchased_and_played_df.groupBy("game").agg(spark_count("user_id").alias("purchased_and_played"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.7 Games Played but Not Purchased
# MAGIC **Here we isolate users who played games without buying them:**
# MAGIC
# MAGIC This could result from free-to-play titles, shared accounts, time-limited demos, or access via bundles.
# MAGIC
# MAGIC This segment helps identify titles with a high trial-to-purchase gap — useful for marketing or monetization strategies.

# COMMAND ----------

# 7. Games played but not purchased
played_not_purchased_df = user_game_plays.join(user_game_purchases, on=["user_id", "game"], how="left_anti")
played_not_purchased_per_game = played_not_purchased_df.groupBy("game").agg(spark_count("user_id").alias("played_not_purchased"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.8 Average Playtime per User for Each Game
# MAGIC **Calculates the mean number of hours spent per user on each game.**
# MAGIC
# MAGIC . Useful for understanding average engagement depth
# MAGIC
# MAGIC . Ideal for comparing casual vs addictive titles

# COMMAND ----------

# 8. Average playtime per user for each game
game_avg_playtime_df = play_only_df \
    .groupBy("game") \
    .agg(spark_avg("value").alias("avg_playtime_per_user"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.9 Maximum Playtime per User for Each Game
# MAGIC **Reveals outlier sessions, users who invested the most time in a game.**
# MAGIC
# MAGIC . Can indicate fanaticism or a deep niche following
# MAGIC
# MAGIC . Important when evaluating long-tail user interest

# COMMAND ----------

# 9. Maximum playtime per user for each game
game_max_playtime_df = play_only_df \
    .groupBy("game") \
    .agg(spark_max("value").alias("max_playtime_per_user"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.10 Minimum Playtime per User for Each Game
# MAGIC **Helps identify:**
# MAGIC
# MAGIC . Titles with many one-and-done sessions
# MAGIC
# MAGIC . Games that may struggle to retain players despite downloads

# COMMAND ----------

# 10. Minimum playtime per user for each game
game_min_playtime_df = play_only_df \
    .groupBy("game") \
    .agg(spark_min("value").alias("min_playtime_per_user"))


# COMMAND ----------

# MAGIC %md
# MAGIC ####3.2.11 Join All Metrics into One Table by Game
# MAGIC **Merges all behavioral metrics into one comprehensive DataFrame:**
# MAGIC
# MAGIC . Purchase metrics (total, 1-time, 2-time)
# MAGIC
# MAGIC . Play metrics (instances, total hours, avg/max/min per user)
# MAGIC
# MAGIC . Binary interaction types (purchase-only, play-only, both)
# MAGIC
# MAGIC . This table is your final game-level analytical feature set — useful for:
# MAGIC
# MAGIC . Profiling user interaction patterns
# MAGIC
# MAGIC . Training interpretable classifiers or clustering models
# MAGIC
# MAGIC . Feeding into dashboards or visualization tools

# COMMAND ----------

# 11. Join all metrics into one table by game
game_metrics_df = purchase_counts_df \
    .join(play_instance_df, on="game", how="outer") \
    .join(play_hours_df, on="game", how="outer") \
    .join(game_avg_playtime_df, on="game", how="outer") \
    .join(game_max_playtime_df, on="game", how="outer") \
    .join(game_min_playtime_df, on="game", how="outer") \
    .join(once_per_game_df, on="game", how="outer") \
    .join(twice_per_game_df, on="game", how="outer") \
    .join(purchase_not_played_per_game, on="game", how="outer") \
    .join(purchased_and_played_per_game, on="game", how="outer") \
    .join(played_not_purchased_per_game, on="game", how="outer")

display(game_metrics_df.orderBy(col("total_purchase").desc_nulls_last()))


# COMMAND ----------

# MAGIC %md
# MAGIC **important**
# MAGIC
# MAGIC as evident from the table, there are no game titles that were played without purchase, meaning, no game were shared or released as demo.

# COMMAND ----------

# MAGIC %md
# MAGIC **example**
# MAGIC
# MAGIC **Dota 2** is the top title with:
# MAGIC
# MAGIC most purchases: 4,841
# MAGIC
# MAGIC most play instances: 4841
# MAGIC
# MAGIC most total hours played: 981,000 
# MAGIC
# MAGIC An average playtime of ~202 hours per user
# MAGIC
# MAGIC A max session length of 10,442 hours
# MAGIC
# MAGIC always played when purchased, always purchased once

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.3 User-Level Interaction Distribution
# MAGIC **This section quantifies how broadly individual users engage with the Steam catalog, a critical factor for:**
# MAGIC
# MAGIC . Cold-start user detection
# MAGIC
# MAGIC . Identifying power users vs one-time players
# MAGIC
# MAGIC . Building appropriate ALS regularization strategies

# COMMAND ----------

# MAGIC %md
# MAGIC **This section:**
# MAGIC
# MAGIC Extracts a deduplicated matrix of user_id–game pairs from the cleaned dataset. Each row represents a unique interaction, regardless of action type (play or purchase). This forms the basis of the implicit interaction matrix used in recommender modeling.
# MAGIC
# MAGIC then, for each user, counts how many unique games they have engaged with. This provides: Breadth of user interest, A proxy for user activeness, and Basis for user segmentation.
# MAGIC
# MAGIC This bins users into fixed-width ranges based on how many games they've interacted with. For example:
# MAGIC
# MAGIC 1–4 → 0
# MAGIC
# MAGIC 5–9 → 5
# MAGIC
# MAGIC 10–14 → 10
# MAGIC ...and so on.
# MAGIC
# MAGIC
# MAGIC afterwards, the **summary statistics** computes core descriptive statistics:
# MAGIC
# MAGIC . Average number of games per user
# MAGIC
# MAGIC . Maximum games any one user interacted with
# MAGIC
# MAGIC . Minimum games (often = 1, indicating cold-start users)
# MAGIC
# MAGIC **These metrics help frame the sparsity profile of the dataset.**
# MAGIC
# MAGIC **lastly**, it Counts how many users fall into each interaction bin and computes the percentage of total users represented in each group. This is essential for:
# MAGIC
# MAGIC . Understanding user engagement skew
# MAGIC
# MAGIC . Designing personalized vs popular-item strategies

# COMMAND ----------

user_game_df = p_steam_df.select("user_id", "game").distinct()

# Generate user-game interaction counts
user_game_interaction_counts = user_game_df \
    .groupBy("user_id") \
    .agg(spark_count("game").alias("num_games_interacted"))

# Binning users by number of games interacted
user_game_bins = user_game_interaction_counts.withColumn(
    "binned_games", (col("num_games_interacted") / 5).cast("int") * 5
)

# Summary statistics on number of games per user
interaction_stats = user_game_interaction_counts.select(
    spark_avg("num_games_interacted").alias("avg_games_per_user"),
    spark_max("num_games_interacted").alias("max_games_per_user"),
    spark_min("num_games_interacted").alias("min_games_per_user")
)
display(interaction_stats)

# Define global interaction stats once and reuse
actual_interactions = user_game_df.count()

from pyspark.sql.functions import lit

# Show percentage of users in each binned interaction group
user_game_bins_summary = user_game_bins.groupBy("binned_games") \
    .agg(spark_count("user_id").alias("user_count")) \
    .withColumn("percentage", (col("user_count") / lit(unique_users)) * 100) \
    .orderBy("binned_games")
display(user_game_bins_summary)


# COMMAND ----------

# MAGIC %md
# MAGIC This clearly reflects a long-tail user behavior distribution: A small number of users engages broadly with many games with the majority (significant) of users only interact with a handful of titles, **mostly, less than 5 gmaes**.
# MAGIC
# MAGIC when pair it with a single user having a 1068 purchases, this also shows most of the purchase is by power users.
# MAGIC
# MAGIC These mean:
# MAGIC
# MAGIC **ALS must handle extreme sparsity — most users provide feedback for fewer than 5 titles.**
# MAGIC
# MAGIC **Regularization must balance power users' influence to avoid model skewing.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 Sparsity
# MAGIC In this section, we calculate the sparsity of the user–game interaction matrix, a vital metric in collaborative filtering. Sparsity quantifies how many actual interactions exist relative to the total possible (i.e., all users rating all games).

# COMMAND ----------

# MAGIC %md
# MAGIC **Matrix Sparsity Calculation**
# MAGIC . unique_users: Number of distinct user IDs (12,393)
# MAGIC
# MAGIC . unique_games: Number of distinct games (5,155)
# MAGIC
# MAGIC . actual_interactions: Observed user–game pairs from the dataset (user_game_df.count())
# MAGIC
# MAGIC . The total possible interactions are calculated as a Cartesian product

# COMMAND ----------

# Calculate sparsity of the user-game matrix
total_possible = unique_users * unique_games
sparsity = 1 - (actual_interactions / total_possible)

print(f"Total Unique Users: {unique_users}")
print(f"Total Unique Games: {unique_games}")
print(f"Actual User-Game Interactions: {actual_interactions}")
print(f"Total Possible Interactions: {total_possible}")
print(f"Matrix Sparsity: {round(sparsity * 100, 4)}%")

# COMMAND ----------

# MAGIC %md
# MAGIC **The matrix is 99.79% sparse, meaning fewer than 0.2% of user–game pairs have observed interactions.**
# MAGIC
# MAGIC This is typical of implicit feedback domains, especially for platforms like Steam, where users play only a small subset of the catalog.

# COMMAND ----------

# MAGIC %md
# MAGIC therefore, **Matrix factorization (e.g., ALS) is ideal for handling sparsity.**
# MAGIC
# MAGIC Also:
# MAGIC
# MAGIC . Regularization and rank selection are critical to balance overfitting from power users and underfitting cold-start users.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Correlation Checks Between Interaction Types
# MAGIC **Understanding whether purchase frequency aligns with play behavior is critical:**
# MAGIC
# MAGIC . High correlation suggests purchases are a proxy for true interest.
# MAGIC
# MAGIC . Low correlation may signal games bought but not played.

# COMMAND ----------

# Importing libraries
from pyspark.sql.functions import corr, col
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.1 Correlation: Total Purchases vs Play Instances per Game
# MAGIC Joins two datasets: **total purchases per game** and **total play sessions per game**.
# MAGIC
# MAGIC . Computes Pearson correlation (corr) between them.
# MAGIC
# MAGIC **This assesses whether popular purchases are also heavily played.**

# COMMAND ----------

# MAGIC %md
# MAGIC **Scatterplot** Converts the Spark DataFrame to pandas for local visualization. it helps visually assess distribution shape and outliers, complementing the correlation score.
# MAGIC
# MAGIC

# COMMAND ----------

# 1. Total purchase vs play instances per game
purchase_play_instance_df = purchase_counts_df \
    .join(play_instance_df, on="game", how="inner")
purchase_play_instance_corr = purchase_play_instance_df \
    .select(corr("total_purchase", "total_play_instance").alias("purchase_play_instance_corr"))
display(purchase_play_instance_corr)

# Scatterplot
pd1 = purchase_play_instance_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(pd1["total_purchase"], pd1["total_play_instance"], alpha=0.5)
plt.xlabel("Total Purchases")
plt.ylabel("Total Play Instances")
plt.title("Purchases vs Play Instances per Game")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **The Pearson correlation between total_purchase and total_play_instance is 0.9495.**
# MAGIC
# MAGIC . The scatterplot clearly shows a linear upward trend; validating that games with high purchase counts tend to have proportionally high play session counts.
# MAGIC
# MAGIC **This further supports your decision to use either purchase or play as a behavioral signal in the implicit feedback matrix.** The pattern holds even for high-outlier titles like "Dota 2" and "CS:GO", which cluster far above the others on both axes.

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.2 Correlation: Total Purchases vs Total Play Hours per Game
# MAGIC This correlation checks whether the number of purchases aligns with the intensity of user engagement, as measured by total hours played.

# COMMAND ----------

# MAGIC %md
# MAGIC the code below, Joins per-game purchase and total play hour summaries.
# MAGIC
# MAGIC . Uses corr() to compute Pearson correlation between purchase count and total hours played.
# MAGIC
# MAGIC **The aim is to verify whether purchase volume translates into actual usage.**

# COMMAND ----------

# 2. Total purchase vs total play hours per game
purchase_play_hours_df = purchase_counts_df \
    .join(play_hours_df, on="game", how="inner")
purchase_play_hours_corr = purchase_play_hours_df \
    .select(corr("total_purchase", "total_play_hours").alias("purchase_play_hours_corr"))
display(purchase_play_hours_corr)

# Scatterplot
pd2 = purchase_play_hours_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(pd2["total_purchase"], pd2["total_play_hours"], alpha=0.5)
plt.xlabel("Total Purchases")
plt.ylabel("Total Play Hours")
plt.title("Purchases vs Play Hours per Game")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Pearson Correlation: **0.8017**. This is a strong positive correlation, though slightly lower than the purchases vs play instances correlation (which was ~0.95). also, the plot shows an upward trend with heavier variance among high-purchase titles.
# MAGIC
# MAGIC **This suggests that while purchases is correlated with play frequency well, total hours can vary more widely.** therefore, **play hour data** might enhance ranking quality in recommender output beyond raw counts.

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.3 Correlation: Total Play Instances vs Total Play Hours per Game
# MAGIC This correlation examines whether games that are played frequently are also played for longer durations, validating session frequency as a proxy for cumulative engagement.
# MAGIC
# MAGIC the code below Joins total play hours and total play sessions per game and Computes Pearson correlation between the two.
# MAGIC
# MAGIC **This captures the frequency-to-duration relationship. do players who log in more often spend more time, or are they logging many short sessions?**

# COMMAND ----------

# 3. Total play hours vs play instances per game
play_hours_instance_df = play_hours_df \
    .join(play_instance_df, on="game", how="inner")
play_hours_instance_corr = play_hours_instance_df \
    .select(corr("total_play_hours", "total_play_instance").alias("play_hours_instance_corr"))
display(play_hours_instance_corr)

# Scatterplot
pd3 = play_hours_instance_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(pd3["total_play_instance"], pd3["total_play_hours"], alpha=0.5)
plt.xlabel("Total Play Instances")
plt.ylabel("Total Play Hours")
plt.title("Play Instances vs Play Hours per Game")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC the Pearson Correlation (0.9013) confirms a very strong positive relationship:
# MAGIC
# MAGIC **Games with many play sessions also accumulate high total hours which Indicates session frequency is associated with play depth.**
# MAGIC
# MAGIC The plot shows a clean linear progression with a few games standing far above others on both axes, as expected from massively multiplayer or competitive games.

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.4 Per User: Purchases vs Total Play Hours
# MAGIC This correlation evaluates whether users who buy more games also tend to play longer. It reflects user engagement alignment between ownership and actual usage.
# MAGIC
# MAGIC the code Aggregates total purchases and total play hours per user, Joins these stats and computes Pearson correlation. it **Measures if users who buy more games also play more.**

# COMMAND ----------

# 4. Per user: purchase vs total play hours
user_purchase_df = purchase_only_df.groupBy("user_id").agg(spark_sum("value").alias("user_total_purchases"))
user_play_hours_df = play_only_df.groupBy("user_id").agg(spark_sum("value").alias("user_total_play_hours"))

user_purchase_play_df = user_purchase_df \
    .join(user_play_hours_df, on="user_id", how="inner")
user_purchase_play_corr = user_purchase_play_df \
    .select(corr("user_total_purchases", "user_total_play_hours").alias("user_purchase_play_hours_corr"))
display(user_purchase_play_corr)

# Scatterplot
pd4 = user_purchase_play_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(pd4["user_total_purchases"], pd4["user_total_play_hours"], alpha=0.5)
plt.xlabel("User Purchases")
plt.ylabel("User Play Hours")
plt.title("User Purchases vs Play Hours")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This is a **moderate positive correlation**, suggesting:
# MAGIC
# MAGIC **Some users who purchase many games do not necessarily play much while others play extensively despite fewer purchases, possibly focused gamers.**
# MAGIC
# MAGIC Based on plot: **data is clustered at low-end** meaning many users with very low purchases and play hours. it is **sparse high-end** showing the power users.
# MAGIC
# MAGIC This validates the presence of diverse usage patterns:
# MAGIC
# MAGIC **. Collectors who buy often but play little**
# MAGIC
# MAGIC **. Core gamers who focus on a few games but invest heavily**

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.5 Per User: Purchases vs Play Instances
# MAGIC This correlation **evaluates whether users who buy more games also engage in more play sessions**. A critical insight into frequency-based engagement patterns.
# MAGIC
# MAGIC The code Joins both **user_total_purchases from earlier section** with **user_play_instances: how many games each user has played (regardless of hours)** on user_id, then computes Pearson correlation between purchase count and number of distinct game interactions.
# MAGIC
# MAGIC

# COMMAND ----------

# 5. Per user: purchase vs play instances
user_play_instances_df = play_only_df.groupBy("user_id").agg(spark_count("game").alias("user_play_instances"))

user_purchase_play_instances_df = user_purchase_df \
    .join(user_play_instances_df, on="user_id", how="inner")
user_purchase_play_instances_corr = user_purchase_play_instances_df \
    .select(corr("user_total_purchases", "user_play_instances").alias("user_purchase_play_instances_corr"))
display(user_purchase_play_instances_corr)

# Scatterplot
pd5 = user_purchase_play_instances_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(pd5["user_total_purchases"], pd5["user_play_instances"], alpha=0.5)
plt.xlabel("User Purchases")
plt.ylabel("Play Instances")
plt.title("User Purchases vs Play Instances")
plt.grid(True)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC This is **a very strong positive correlation; stronger than purchases vs hours.**
# MAGIC
# MAGIC Users with purchase  also tend to play. However, outliers exist (e.g., highly active players with low purchases or collectors with high purchases but modest play counts).

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.5.6 Per User: Play Hours vs Play Instances
# MAGIC This correlation tests whether users who engage in many play sessions also accumulate high total hours; assessing consistency between frequency and duration of interaction.
# MAGIC
# MAGIC The code **joins play instance counts and total play hours at the user level** then measures how often users play vs how long they play — per session engagement density.

# COMMAND ----------

# 6. Per user: play hours vs play instances
user_play_behavior_df = user_play_instances_df.join(user_play_hours_df, on="user_id", how="inner")
user_play_behavior_corr_df = user_play_behavior_df \
    .select(corr("user_play_instances", "user_total_play_hours").alias("user_play_hours_vs_instances_corr"))
display(user_play_behavior_corr_df)

# Scatterplot
user_play_behavior_pd = user_play_behavior_df.toPandas()
plt.figure(figsize=(6,4))
plt.scatter(user_play_behavior_pd["user_play_instances"], user_play_behavior_pd["user_total_play_hours"], alpha=0.5)
plt.xlabel("Play Instances")
plt.ylabel("Play Hours")
plt.title("User Play Instances vs Play Hours")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This is a **moderate correlation**; significantly weaker than other user-level metrics.
# MAGIC
# MAGIC **This indicates high behavioral variance:**
# MAGIC
# MAGIC . Some users play frequently but in short bursts
# MAGIC
# MAGIC . Others play infrequently but for long periods
# MAGIC
# MAGIC The Visual Pattern also shows dense cluster at low values (most users)

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.6 Top Games and Users by Engagement Metrics
# MAGIC This section summarizes the most influential content and consumers in the dataset, based on:
# MAGIC
# MAGIC . Total purchases
# MAGIC
# MAGIC . Total play hours
# MAGIC
# MAGIC . Total play instances
# MAGIC
# MAGIC **This allows for:**
# MAGIC
# MAGIC . Popularity-based priors
# MAGIC
# MAGIC . Power-user analysis
# MAGIC
# MAGIC . Outlier detection and behavior profiling

# COMMAND ----------

# MAGIC %md
# MAGIC **First**, the code retrieves and plots the top 10 games across three axes of engagement (purchase, play instance and play hours). it uses horizontal bar charts for readability to enable cross-comparison between different behavior dimensions. 
# MAGIC
# MAGIC **Next,** each user's total purchase count and total hours played are calculated from the filtered action-specific DataFrames. **spark_sum("value")** is used because **for purchases, values were summed from value = 1 entries and For play, value represents playtime in hours.**
# MAGIC
# MAGIC **lastly**, their results are ploted.

# COMMAND ----------

# Top 10 Games by Purchase, Play Hours, and Play Instances
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

# Convert to pandas for visualization
purchase_top10_pd = purchase_counts_df.orderBy(col("total_purchase").desc()).limit(10).toPandas()
play_hours_top10_pd = play_hours_df.orderBy(col("total_play_hours").desc()).limit(10).toPandas()
play_instance_top10_pd = play_instance_df.orderBy(col("total_play_instance").desc()).limit(10).toPandas()

# Plot side-by-side bar charts
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].barh(purchase_top10_pd["game"], purchase_top10_pd["total_purchase"], color="steelblue")
axs[0].set_title("Top 10 Games by Purchases")
axs[0].invert_yaxis()

axs[1].barh(play_hours_top10_pd["game"], play_hours_top10_pd["total_play_hours"], color="darkorange")
axs[1].set_title("Top 10 Games by Play Hours")
axs[1].invert_yaxis()

axs[2].barh(play_instance_top10_pd["game"], play_instance_top10_pd["total_play_instance"], color="forestgreen")
axs[2].set_title("Top 10 Games by Play Instances")
axs[2].invert_yaxis()

plt.tight_layout()
plt.show()

# Ensure user-level totals exist
user_purchase_df = purchase_only_df.groupBy("user_id").agg(spark_sum("value").alias("user_total_purchases"))
user_play_hours_df = play_only_df.groupBy("user_id").agg(spark_sum("value").alias("user_total_play_hours"))

# Top 10 Users by Purchases and Play Hours
user_purchase_top10_pd = user_purchase_df.orderBy(col("user_total_purchases").desc()).limit(10).toPandas()
user_play_hours_top10_pd = user_play_hours_df.orderBy(col("user_total_play_hours").desc()).limit(10).toPandas()

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].barh(user_purchase_top10_pd["user_id"].astype(str), user_purchase_top10_pd["user_total_purchases"], color="mediumpurple")
axs[0].set_title("Top 10 Users by Purchases")
axs[0].invert_yaxis()

axs[1].barh(user_play_hours_top10_pd["user_id"].astype(str), user_play_hours_top10_pd["user_total_play_hours"], color="tomato")
axs[1].set_title("Top 10 Users by Play Hours")
axs[1].invert_yaxis()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  Results show: **Dota 2 ranks #1 across all three metrics, indicating unmatched depth and breadth.**
# MAGIC
# MAGIC **Some games (e.g., Unturned) perform better on purchase and play than play hours suggesting they may not be as engaging.**
# MAGIC
# MAGIC Also, top users are behaviorally diverse because **top purchase users are different from top play hour users, highlighting collector vs committed gamer distinctions.**
# MAGIC
# MAGIC Therefore, **ALS regularization should not assume every buyer is also a power user.**

# COMMAND ----------

# MAGIC %md
# MAGIC ##****Choosing the METRIC****
# MAGIC
# MAGIC **Play hours** is the only variable in the dataset that is **continuous** (e.g., 0.2 hrs to 10,000+ hrs), **Reflects intensity**, not just frequency, and **Represents how much a user truly engaged with a game**. It is also **alignes well** and adds depth to **frequency-based metrics like purchases or play counts**.
# MAGIC
# MAGIC Additionally, **ALS works best when the confidence matrix reflects intensity, which only** *Play hours* **captures across the full user spectrum.**
# MAGIC
# MAGIC **Purchases and Play Instances are Easily Inflated**. Users may buy bundles during Steam sales leading to high purchase count, no intent.
# MAGIC
# MAGIC **Cheap games** can have high play instances but minimal depth.
# MAGIC
# MAGIC **Only play hours:**
# MAGIC
# MAGIC . Cannot be faked or “accidental”
# MAGIC
# MAGIC . Represents actual user effort and retention
# MAGIC
# MAGIC As a result, **Play hours** aligns most closely with true satisfaction, the actual target of a recommender system.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Recommender system -  Alternating Least Squares (ALS)
# MAGIC This section implements a collaborative filtering model using PySpark’s ALS (Alternating Least Squares) for implicit feedback. The model learns latent user and item features using play hours as the rating proxy, shown in earlier analysis to be the most behaviorally consistent engagement signal.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 ALS on all of the dataset with no filtering
# MAGIC The model is trained on the entire dataset. This maximizes usage of all available user–game interaction signals, particularly focusing on:
# MAGIC
# MAGIC **. Playtime (value)(log-transformed play hours)**

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.1

# COMMAND ----------

# MAGIC %md
# MAGIC **StringIndexer** is used to convert string-based user_id and game names into unique integers, required for matrix construction.
# MAGIC
# MAGIC **log1p()** transforms raw play hours to reduce skew from extreme values. Raw play hours can span from 1 hr to 10,000+ hrs, creating instability. log1p compresses extreme values while preserving order and makes ALS more robust to outliers.
# MAGIC
# MAGIC **value** is already filtered to only include valid playtime records (from EDA cleaning). This forms the interaction matrix (rows = users, columns = games, values = hours played).
# MAGIC
# MAGIC **Indexing** transforms string IDs into numerical matrix coordinates which enables the ALS algorithm to build latent factor matrices.
# MAGIC
# MAGIC **display** shows the transformed data in a table.

# COMMAND ----------

# Step 1: Assign Integer IDs for ALS (user_id and game)
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import log1p

# Use filtered play_only_df as input for ALS
als_input_raw_df = play_only_df.select("user_id", "game", "value")

# Apply StringIndexer to user_id and game
user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_index")
game_indexer = StringIndexer(inputCol="game", outputCol="game_index")

# Fit and transform
user_indexed_model = user_indexer.fit(als_input_raw_df)
game_indexed_model = game_indexer.fit(als_input_raw_df)

als_indexed_df = user_indexed_model.transform(als_input_raw_df)
als_indexed_df = game_indexed_model.transform(als_indexed_df)

# Step 2: Normalize play hours using log1p to reduce skew
als_indexed_df = als_indexed_df.withColumn("rating", log1p(col("value")))

# Preview ALS-ready DataFrame
display(als_indexed_df.select("user_id", "game", "user_id_index", "game_index", "value", "rating"))


# COMMAND ----------

# MAGIC %md
# MAGIC **Ratings** reflect confidence in user preference:
# MAGIC
# MAGIC . Short sessions yield low ratings (close to 0) and Heavy sessions (hundreds of hours) produce high ratings.
# MAGIC
# MAGIC Here. all users and games are retained, even those with a single interaction. This approach maximizes learning from long-tail users and niche games which is useful as a baseline.later, filtering can be introduced for performance comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.2 Train-Test Split for ALS Model Evaluation
# MAGIC Before training the ALS recommender model, the dataset is randomly split into training (80%) and testing (20%) sets. This allows us to evaluate how well ALS generalizes to unseen data.

# COMMAND ----------

# Step 3: Train/Test split for ALS evaluation
(training_df, test_df) = als_indexed_df.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {training_df.count()}")
print(f"Test set size: {test_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.3 Hyperparameter Tuning and MLflow Setup
# MAGIC This section initializes MLflow for tracking model runs and defines the evaluation metric.
# MAGIC
# MAGIC Dynamically identifies the Databricks user and creates a task-scoped MLflow path.
# MAGIC
# MAGIC **autolog()** automatically logs all ALS models, parameters, and metrics for comparison.
# MAGIC
# MAGIC **Root Mean Squared Error** is used to measure accuracy of predictions vs. true log-play hours **because:**
# MAGIC
# MAGIC . The target variable (rating) is continuous (not binary).
# MAGIC
# MAGIC . ALS outputs real-valued predictions, which align with RMSE evaluation.
# MAGIC
# MAGIC **the users' notebook address is also extracted so multiple users can run the code**

# COMMAND ----------

# Step 4: Hyperparameter tuning and MLflow tracking
from pyspark.ml.recommendation import ALS
import mlflow
import mlflow.spark
from pyspark.ml.evaluation import RegressionEvaluator

# Get the current user's email from the notebook context
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Create experiment paths using the current user
experiment_path = f"/Users/{current_user}/Task 2"

# Set up MLflow experiments
mlflow.set_experiment(experiment_path)
mlflow.pyspark.ml.autolog()

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.4 ALS Model Training and Grid Search with MLflow
# MAGIC This cell runs a hyperparameter grid search to find the optimal values of:
# MAGIC
# MAGIC **rank:** number of latent factors (dimensionality of user/game embeddings)
# MAGIC
# MAGIC **regParam:** regularization strength to prevent overfitting
# MAGIC
# MAGIC **6 combination** are evaluated using Root Mean Squared Error (RMSE) on the test set.

# COMMAND ----------

# MAGIC %md
# MAGIC The **code** iterates over all combinations of:
# MAGIC
# MAGIC rank - {5, 10, 15}
# MAGIC
# MAGIC regParam - {0.05, 0.1}
# MAGIC
# MAGIC Starts a separate MLflow run for each configuration.
# MAGIC
# MAGIC Then **trains ALS model on training_df, applies it to test_df to predict log-play hours, and Uses RegressionEvaluator to compute RMSE.**
# MAGIC
# MAGIC **log_param**	Records hyperparameter values
# MAGIC
# MAGIC **log_metric("rmse")**	Tracks model performance
# MAGIC
# MAGIC **log_model()**	Saves the actual trained model (reproducibility & inference ready)
# MAGIC
# MAGIC **Drops cold-start predictions to avoid noisy RMSE scores from unseen users or games**

# COMMAND ----------

# Run ALS for multiple combinations of rank and regParam
for rank in [5, 10, 15]:
    for reg in [0.05, 0.1]:
        with mlflow.start_run():
            als = ALS(
                userCol="user_id_index",
                itemCol="game_index",
                ratingCol="rating",
                implicitPrefs=True,
                coldStartStrategy="drop",
                nonnegative=True,
                maxIter=10,
                rank=rank,
                regParam=reg
            )

            als_model = als.fit(training_df)
            predictions = als_model.transform(test_df)
            rmse = evaluator.evaluate(predictions)
            print(f"Rank={rank}, RegParam={reg} → RMSE: {rmse:.4f}")

            mlflow.log_param("rank", rank)
            mlflow.log_param("regParam", reg)
            mlflow.log_metric("rmse", rmse)
            mlflow.spark.log_model(als_model, "als_model")

# COMMAND ----------

# MAGIC %md
# MAGIC Best Performing Configuration **before checking the MLFlow**:
# MAGIC
# MAGIC Rank = 15
# MAGIC
# MAGIC RegParam = 0.1
# MAGIC
# MAGIC Lowest RMSE = 2.3154
# MAGIC
# MAGIC This combination indicates that higher latent dimensionality (15) and slightly higher regularization (0.1) produced the best generalization performance.
# MAGIC
# MAGIC **As a result:**
# MAGIC
# MAGIC Lower RMSE = Better fit on unseen data, meaning better personalized recommendations. we can see consistent improvements as we increase the rank (complexity of latent factor interactions). Regularization helps control overfitting; RMSE slightly improved when moving from 0.05 to 0.1.

# COMMAND ----------

# MAGIC %md
# MAGIC ****Now, we track the recommender system using MLflow****
# MAGIC
# MAGIC based on the loggs, the Model **"thoughtful-auk-928"** is the best choice because its **RMSE: 2.3061**. This is the best model from the runs.
# MAGIC it uses **rank(15) and regParam(0.05)**.
# MAGIC
# MAGIC MLflow improves reliability, reproducibility, and transparency of your model training pipeline. It not only helps identify the best ALS model objectively using RMSE, but also packages everything needed to validate, register, and deploy the model.
# MAGIC Hence, for this ALS recommender system, **MLflow-selected model is more accurate and production-ready than selecting manually from notebook outputs.**
# MAGIC
# MAGIC Therefore, we choose the **model ID (model_uri)** for this model **from mlflow** and continue the analysis.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.5 Load Best Model from MLflow & Predict
# MAGIC it Loads the best-performing ALS model (lowest RMSE) from MLflow by referencing its unique run ID and artifact path.
# MAGIC
# MAGIC **mlflow.spark.load_model()** pulls the model into the current notebook session. 
# MAGIC
# MAGIC Then applies the trained ALS model to the test dataset and **generates new predictions** for the “rating” (user’s preference or interaction strength with a game).
# MAGIC
# MAGIC It computes the Root Mean Squared Error (RMSE) between actual and predicted ratings on test data and displays a table showing:
# MAGIC
# MAGIC **. user_id_index:** Internal index representing a user.
# MAGIC
# MAGIC **. game_index:** Internal index representing a game.
# MAGIC
# MAGIC **. rating:** Log-normalized actual value (e.g. play hours).
# MAGIC
# MAGIC **. prediction:** ALS-predicted rating from the model.

# COMMAND ----------

# Load the best model from MLflow manually selected best model URI
model_uri = "runs:/e73672b2b4384b9fb19c4fbea6b0cd66/als_model"  # Replace with your best model URI
loaded_model = mlflow.spark.load_model(model_uri)
predictions = loaded_model.transform(test_df)
display(predictions.select("user_id_index", "game_index", "rating", "prediction"))
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC For users with high actual ratings (e.g. 4.3, 3.8), predicted values are significantly lower. Also, ALS prediction scale is compressed compared to the normalized rating values.
# MAGIC
# MAGIC **. This is expected. ALS predicts relative preferences rather than exact magnitudes.**
# MAGIC
# MAGIC The model correctly distinguishes low-rated interactions with low predictions (e.g. 0.03, 0.09).

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.6 – Additional Evaluation: MAE and Residual Analysis
# MAGIC The code imports key aggregation and transformation functions for Spark SQL operations.
# MAGIC
# MAGIC **MAE (Mean Absolute Error)** is then calculated using .agg(avg(...)). This provides a scale-aware, non-squared measure of prediction error. **It complements RMSE by being less sensitive to extreme outliers.**
# MAGIC
# MAGIC **Here, MAE shows that on average, the model’s predicted play hours differ from the actual normalized play hours by approximately 1.75 units.**
# MAGIC
# MAGIC The code also adds a new column residual representing the error per row which is useful for understanding where the model performs poorly (outliers). This helps target improvement by allowing to flag cases where game popularity skews prediction.

# COMMAND ----------

# Additional Evaluation MAE
from pyspark.sql.functions import abs as spark_abs, avg as spark_avg, pow, sqrt, explode
import matplotlib.pyplot as plt

mae = predictions.withColumn("abs_error", spark_abs(col("rating") - col("prediction"))).agg(spark_avg("abs_error")).first()[0]
print(f"MAE: {mae:.4f}")



residuals_df = predictions.withColumn("residual", spark_abs(col("rating") - col("prediction")))
display(residuals_df.orderBy("residual", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the table, the biggest residuals occur for very high play hour users, especially with Dota 2.
# MAGIC
# MAGIC This indicates:
# MAGIC
# MAGIC **Heavy users are not well-predicted by ALS in absolute terms.**
# MAGIC
# MAGIC **ALS likely averages or smooths across most users, underfitting outliers.**
# MAGIC
# MAGIC **Dota 2 repeatedly appears, probably the most played game, but hard to model accurately.**
# MAGIC
# MAGIC **Top residuals** indicate the model struggles with highly active users and very popular games like Dota 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.7 Actual vs Predicted Ratings Plot
# MAGIC The provided code is aimed at generating a scatter plot for visual comparison between the actual and predicted ratings. It extracts the **rating (actual)** and **prediction** columns into a pandas DataFrame (pred_pd) for visualization.

# COMMAND ----------

# Actual vs Predicted Plot
pred_pd = predictions.select("rating", "prediction").toPandas()
plt.figure(figsize=(6, 6))
plt.scatter(pred_pd["rating"], pred_pd["prediction"], alpha=0.3)
plt.xlabel("Actual Rating (log1p hours)")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")
plt.grid(True)
plt.show()

print("Predictions made:", predictions.count())
print("Total test set size:", test_df.count())


# COMMAND ----------

# MAGIC %md
# MAGIC **Based on the scatter plot**,
# MAGIC
# MAGIC A high density of **predictions** is crowded below predicted rating ≈ 1.0. This reflects a limitation of the ALS model where predictions are compressed near the upper boundary, often **failing to represent high variance in actual ratings.**
# MAGIC
# MAGIC For **actual ratings** greater than ~5, many predictions plateau near 0.8–1.0.
# MAGIC
# MAGIC **This confirms the model’s average predictions are close enough for general trends,** as shown by the MAE and RMSE scores. But the prediction saturation near 1.0 is **a sign of ALS limitations, likely due to cold start drop strategy or sparse interaction levels.**

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.1.8: Per-User RMSE Analysis
# MAGIC In this section, we dive deeper into model performance on a user-by-user basis using RMSE (Root Mean Squared Error), one of the most reliable regression metrics. Then the histogram of per-user RMSE values helps understand error distribution across the user population.

# COMMAND ----------

# Per-user RMSE
user_rmse_df = predictions.withColumn("squared_error", pow(col("rating") - col("prediction"), 2)) \
    .groupBy("user_id_index").agg(sqrt(spark_avg("squared_error")).alias("user_rmse"))
display(user_rmse_df.orderBy("user_rmse", ascending=False))

user_rmse_pd = user_rmse_df.toPandas()
plt.figure(figsize=(8, 5))
plt.hist(user_rmse_pd["user_rmse"], bins=30, color='skyblue', edgecolor='black')
plt.xlabel("User RMSE")
plt.ylabel("Frequency")
plt.title("Distribution of Per-User RMSE")
plt.grid(axis='y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The histogram and table show most users have RMSE between 0 and 1, with the distribution heavily right-skewed, **indicating a majority of users are well-predicted, but a smaller group sees larger errors.**
# MAGIC
# MAGIC **This long tail (users with RMSE above 5–6) could be:**
# MAGIC
# MAGIC . Sparse interaction histories.
# MAGIC
# MAGIC . Irregular play patterns.
# MAGIC
# MAGIC . Cold-start scenarios (new users).

# COMMAND ----------

# MAGIC %md
# MAGIC #### **4.1.9 Game Recommendation Using the Model**
# MAGIC This block executes the final step in building a practical recommendation system; Top-5 game recommendation for each user using the trained ALS model.
# MAGIC
# MAGIC It checks whether the loaded model is part of a pipeline **(hasattr(..., 'stages'))**. Then calls **.recommendForAllUsers(5)** to generate top 5 **personalized game predictions per user.**

# COMMAND ----------

# Generate Top-5 Recommendations
if hasattr(loaded_model, 'stages'):
    als_model = loaded_model.stages[-1]
    user_recommendations_df = als_model.recommendForAllUsers(5)
else:
    user_recommendations_df = loaded_model.recommendForAllUsers(5)

display(user_recommendations_df)

game_mapping_df = als_indexed_df.select("game", "game_index").distinct()
recommendations_exploded_df = user_recommendations_df.withColumn("rec", explode("recommendations"))
recommendations_flat_df = recommendations_exploded_df.select(
    col("user_id_index"),
    col("rec.game_index").alias("game_index"),
    col("rec.rating").alias("predicted_rating")
)
final_recommendations_df = recommendations_flat_df.join(game_mapping_df, on="game_index", how="left")
display(final_recommendations_df.orderBy("user_id_index", "predicted_rating", ascending=[True, False]))


# COMMAND ----------

# MAGIC %md
# MAGIC This format shows a compact representation of all top-5 recommendations as arrays for each user. It is useful when exporting to applications or storing data where detailed game IDs and predicted ratings per user are needed in nested format.
# MAGIC
# MAGIC it also helps us understand which game-user combinations have the highest predicted affinity.
# MAGIC
# MAGIC Notably, games like Garry's Mod appear repeatedly, which may suggest it has wide appeal across different user clusters.
# MAGIC
# MAGIC **Recommendations vary  across users, which implies the model successfully personalized outcomes.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 ALS on filtered dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.1 User Filtering
# MAGIC The code performs user filtering based on engagement threshold (number of distinct games a user has interacted with). This is a common data quality filtering step to improve model training and prediction reliability in collaborative filtering systems like ALS.
# MAGIC
# MAGIC **We filter to user with more than 5 game interactions because:**
# MAGIC
# MAGIC Users with very few interactions provide weak signals to the ALS model.
# MAGIC
# MAGIC Focusing on “active” users improves the quality of latent factors the model learns.
# MAGIC
# MAGIC New or inactive users might bias the matrix with random or inconsistent preferences.
# MAGIC
# MAGIC

# COMMAND ----------

# Count distinct games per user from the already indexed dataset
user_game_count_df = als_indexed_df.groupBy("user_id_index").agg(countDistinct("game_index").alias("game_count"))

# Find users with more than 5 game interactions
active_users_df = user_game_count_df.filter(col("game_count") > 5)

# Get statistics about the filtered users
active_users_count = active_users_df.count()
total_users_count = user_game_count_df.count()
active_percentage = (active_users_count / total_users_count) * 100

print(f"Users with >5 game interactions: {active_users_count} out of {total_users_count} ({active_percentage:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Only 18.76% of users (2,129) had enough interactions to be considered "active."**
# MAGIC
# MAGIC 81.24% of users were excluded from model training because their data was too sparse. A long-tail distribution where **most users are only marginally engaged, while a small portion drives the majority of activity.**

# COMMAND ----------

# MAGIC %md
# MAGIC By filtering out users with insufficient behavior, the ALS model trained in upcoming steps will:
# MAGIC
# MAGIC **Focus on users who have stronger and more reliable patterns.**
# MAGIC
# MAGIC **Reduce noise in the collaborative filtering matrix.**
# MAGIC
# MAGIC **Likely achieve lower RMSE and MAE scores, which we will verify in subsequent stages.**

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.2 – Filtering and Previewing the Dataset
# MAGIC The code joins the full ALS indexed dataset with the active_users_df, which was filtered before. The result only retains rows corresponding to users who have interacted with >5 games. Then, it calculates and prints the number of interactions in the original vs. filtered dataset.

# COMMAND ----------

# Filter the already indexed dataset to only include active users
filtered_als_indexed_df = als_indexed_df.join(
    active_users_df.select("user_id_index"), 
    on="user_id_index", 
    how="inner"
)

# Show the difference in dataset size
print(f"Original indexed dataset size: {als_indexed_df.count()} interactions")
print(f"Filtered indexed dataset size: {filtered_als_indexed_df.count()} interactions")
print(f"Reduction: {100 - (filtered_als_indexed_df.count() / als_indexed_df.count() * 100):.2f}%")

# COMMAND ----------

# Preview the filtered dataset
display(filtered_als_indexed_df.select("user_id", "game", "user_id_index", "game_index", "value", "rating"))

# COMMAND ----------

# MAGIC %md
# MAGIC **We can clearly see that users retained in this dataframe have multiple game interactions.**
# MAGIC
# MAGIC This ensures the ALS model learns from representative behavior which enhances Model accuracy, Recommendation reliability, and Evaluation consistency.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.3 Dataset Splitting
# MAGIC The code splits dataset into training and test sets

# COMMAND ----------

# Split the filtered dataset using the same seed for reproducibility
(filtered_training_df, filtered_test_df) = filtered_als_indexed_df.randomSplit([0.8, 0.2], seed=42)

print(f"Filtered training set size: {filtered_training_df.count()} interactions")
print(f"Filtered test set size: {filtered_test_df.count()} interactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.4 Creating an ALS Experiment on the Filtered Dataset
# MAGIC In this part we prepare for the training and evaluation of the ALS model using the filtered dataset (which only includes users with >5 game interactions).
# MAGIC
# MAGIC We setup a new MLflow experiment under a separate path so the results from this run are distinct from the earlier full-dataset ALS which **allows for direct comparison between full datset model and filtered datset model.**
# MAGIC
# MAGIC Finaly, we create an evaluator using the same RMSE metric used before.

# COMMAND ----------

# Create a separate experiment for the filtered dataset

# Create experiment paths using the current user
filtered_experiment_path = f"/Users/{current_user}/Task 2 - Filtered Dataset"

# Create a separate experiment for the filtered dataset
mlflow.set_experiment(filtered_experiment_path)
mlflow.pyspark.ml.autolog()

# Use the same evaluator approach
filtered_evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="rating", 
    predictionCol="prediction"
)


# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.5 Training ALS on Filtered Dataset with Hyperparameter Tuning
# MAGIC This part of the notebook runs ALS model training and evaluation using the filtered dataset created earlier, with multiple hyperparameter combinations. It mirrors the full-dataset tuning loop but now focuses on the active user segment only.
# MAGIC
# MAGIC It loops over 6 combinations of **rank**(Number of latent factors (model complexity)) and **regParam**(Regularization to prevent overfitting).
# MAGIC
# MAGIC Then with **mlflow.start_run()** each combination is logged as a separate run in MLflow. This allows easy comparison of RMSE and model parameters.
# MAGIC
# MAGIC It is modeling based on implicit feedback (playtime), not explicit ratings, Skips any test rows with unseen users/games, avoiding NaNs in predictions, Ensures factor values are ≥0, which often improves interpretability.
# MAGIC
# MAGIC Next, the code evaluates accuracy using Root Mean Square Error (RMSE), Logs the hyperparameters and RMSE, and Saves the trained model in MLflow for deployment or comparison.

# COMMAND ----------

# Run ALS for the same hyperparameter combinations
for rank in [5, 10, 15]:
    for reg in [0.05, 0.1]:
        with mlflow.start_run():
            filtered_als = ALS(
                userCol="user_id_index",
                itemCol="game_index",
                ratingCol="rating",
                implicitPrefs=True,
                coldStartStrategy="drop",
                nonnegative=True,
                maxIter=10,
                rank=rank,
                regParam=reg
            )

            filtered_als_model = filtered_als.fit(filtered_training_df)
            filtered_predictions = filtered_als_model.transform(filtered_test_df)
            filtered_rmse = filtered_evaluator.evaluate(filtered_predictions)
            print(f"Filtered dataset - Rank={rank}, RegParam={reg} → RMSE: {filtered_rmse:.4f}")

            mlflow.log_param("rank", rank)
            mlflow.log_param("regParam", reg)
            mlflow.log_metric("rmse", filtered_rmse)
            mlflow.spark.log_model(filtered_als_model, "filtered_als_model")


# COMMAND ----------

# MAGIC %md
# MAGIC Before checking the model from the mlflow, the best c onfiguration is
# MAGIC
# MAGIC **Rank: 10**, and **RegParam: 0.05**, giving the **RMSE: 2.2467**
# MAGIC
# MAGIC **The lowest RMSE (2.2467) indicates the model is making more accurate predictions after filtering out low-interaction users.**
# MAGIC
# MAGIC This aligns with best practice in recommender systems, focusing on denser user histories results in sharper latent factor learning. With **Rank 10**, the model has enough dimensions to capture meaningful variation without overfitting (which may occur at higher ranks). **RegParam 0.05** encourages generalization, striking a solid balance between fit and complexity.

# COMMAND ----------

# MAGIC %md
# MAGIC %md  
# MAGIC ****Now, we track the recommender system using MLflow for the filtered dataset****
# MAGIC
# MAGIC Based on the MLflow logs, the model **"classy-cat-999"** is the best choice, as it achieved the **lowest RMSE: 2.2289** among all runs on the filtered dataset.  
# MAGIC This model is trained with **rank = 15** and **regParam = 0.05**, which represents the optimal hyperparameter combination under the filtered data scenario.
# MAGIC
# MAGIC MLflow greatly enhances the reliability, reproducibility, and transparency of our ALS recommender pipeline. It allows consistent experiment tracking and comparison, making it easier to select the best model objectively using RMSE as a performance metric.
# MAGIC
# MAGIC Additionally, MLflow logs all model dependencies and structure, which makes deployment-ready packaging seamless.  
# MAGIC Hence, **this MLflow-selected model offers superior accuracy and is best suited for production**, compared to selecting a model manually based on notebook outputs alone.
# MAGIC
# MAGIC Therefore, we choose the **model ID (model_uri)** for this run from MLflow and continue our evaluation and deployment using this best model from the filtered dataset.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.6 Final Evaluation of Best Filtered Model based on MLFLOW
# MAGIC This cell finalizes the evaluation of the best model selected from MLflow for the filtered dataset.
# MAGIC
# MAGIC Loads the top-performing ALS model from MLflow (previously tracked as classy-cat-999) using its run URI. This ensures consistent reproduction of the best result from filtered ALS experiments.
# MAGIC
# MAGIC The code applies the ALS model to the test subset of the filtered dataset to obtain predicted ratings. Next, it computes RMSE (Root Mean Squared Error), a standard metric for recommender accuracy. Additionally calculates MAE (Mean Absolute Error) for interpretability. The absolute difference between actual and predicted log1p-hours is averaged for all users.
# MAGIC
# MAGIC RMSE = 2.2289 is excellent for a collaborative filtering system on a large, real-world dataset like Steam.
# MAGIC
# MAGIC MAE = 1.6253 suggests that, on average, predicted log-play-hours deviate by ~1.6 units from actual values, a strong indicator of accurate personalized recommendations.

# COMMAND ----------

# MLflow results manually selected best model URI
best_filtered_model_uri = "runs:/03ca60e187ff4c93b04a0eca292605ad/filtered_als_model"  # Replace with your best filtered model URI
best_filtered_model = mlflow.spark.load_model(best_filtered_model_uri)

# Generate predictions on test set
filtered_predictions = best_filtered_model.transform(filtered_test_df)

# Calculate evaluation metrics
filtered_rmse = filtered_evaluator.evaluate(filtered_predictions)
filtered_mae = filtered_predictions.withColumn(
    "abs_error", 
    spark_abs(col("rating") - col("prediction"))
).agg(spark_avg("abs_error")).first()[0]

print(f"Best Filtered Model - RMSE: {filtered_rmse:.4f}, MAE: {filtered_mae:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.7 Actual vs. Predicted Plot
# MAGIC This cell visualises the predictive performance of the best filtered ALS model.

# COMMAND ----------

# Actual vs Predicted Plot
filtered_pred_pd = filtered_predictions.select("rating", "prediction").toPandas()
plt.figure(figsize=(6, 6))
plt.scatter(filtered_pred_pd["rating"], filtered_pred_pd["prediction"], alpha=0.3)
plt.xlabel("Actual Rating (log1p hours)")
plt.ylabel("Predicted Rating")
plt.title("Filtered Dataset: Actual vs Predicted Ratings")
plt.grid(True)
plt.show()

print("Predictions made:", filtered_predictions.count())
print("Total filtered test set size:", filtered_test_df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC **Dense clusters at the bottom suggest many predictions are conservative or close to the mean.**
# MAGIC
# MAGIC There is some vertical spread at each x-value, but clear upward trend from left to right, confirming the model captures relative preferences.
# MAGIC
# MAGIC **The full dataset included many users with very few interactions, even just one game. These contribute noise, not meaningful preference patterns. In contrast, the filtered dataset focuses on active users, which gives ALS more consistent behaviour patterns to learn from.**
# MAGIC
# MAGIC This leads to better latent feature learning and stronger collaborative signals.

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.8 Per-User RMSE Analysis on Filtered Dataset
# MAGIC This section evaluates how well the ALS model (trained on the filtered dataset) performs on an individual user level, using RMSE (Root Mean Square Error)
# MAGIC
# MAGIC The code This computes the RMSE for each user by calculating the squared error between actual and predicted ratings then aggregating these by user and applying square root of the mean. Lastly, uses matplotlib to plot a histogram of RMSE values across users.

# COMMAND ----------

# Per-user RMSE
filtered_user_rmse_df = filtered_predictions.withColumn(
    "squared_error", 
    pow(col("rating") - col("prediction"), 2)
).groupBy("user_id_index").agg(
    sqrt(spark_avg("squared_error")).alias("user_rmse")
)

# Plot distribution of per-user RMSE
filtered_user_rmse_pd = filtered_user_rmse_df.toPandas()
plt.figure(figsize=(8, 5))
plt.hist(filtered_user_rmse_pd["user_rmse"], bins=30, color='lightgreen', edgecolor='black')
plt.xlabel("User RMSE")
plt.ylabel("Frequency")
plt.title("Distribution of Per-User RMSE (Filtered Dataset)")
plt.grid(axis='y')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The peak is around RMSE 2.0, showing that the majority of users receive reasonably accurate recommendations.
# MAGIC
# MAGIC The distribution is somewhat bell-shaped, indicating stable prediction performance across users.
# MAGIC
# MAGIC Some outliers exist (users with RMSE > 6), but they are rare and not skewing the model significantly.
# MAGIC
# MAGIC **So far, Filtered ALS model provides consistently lower and more centered per-user RMSE.**
# MAGIC
# MAGIC **This confirms its superior reliability and personalization accuracy compared to the model trained on the unfiltered dataset.**
# MAGIC
# MAGIC **It is a strong indicator that filtering users with fewer than 5 interactions leads to a higher quality recommendation system.**

# COMMAND ----------

# MAGIC %md
# MAGIC ####4.2.9 Displaying Recommendations from the Best Filtered ALS Model
# MAGIC This final code cell in the ALS pipeline executes and displays personalized Top-5 game recommendations for each user using the best model selected via MLflow from the filtered dataset.
# MAGIC
# MAGIC If the loaded ALS model is part of a PipelineModel, we access its last stage ([-1]) to get the trained ALS model; otherwise, use the best_filtered_model directly.
# MAGIC
# MAGIC **recommendForAllUsers(5)** Produces a list of 5 recommended games per user based on the model’s predicted rating.
# MAGIC
# MAGIC Finaly, the recommendations are shown in a table

# COMMAND ----------

# Generate recommendations for all users
if hasattr(best_filtered_model, 'stages'):
    filtered_als_model = best_filtered_model.stages[-1]
    filtered_user_recommendations_df = filtered_als_model.recommendForAllUsers(5)
else:
    filtered_user_recommendations_df = best_filtered_model.recommendForAllUsers(5)

# Map game indices back to game names using the existing mapping
game_mapping_df = filtered_als_indexed_df.select("game", "game_index").distinct()

# Expand the recommendations array
filtered_recommendations_exploded_df = filtered_user_recommendations_df.withColumn(
    "rec", 
    explode("recommendations")
)

# Flatten the structure
filtered_recommendations_flat_df = filtered_recommendations_exploded_df.select(
    col("user_id_index"),
    col("rec.game_index").alias("game_index"),
    col("rec.rating").alias("predicted_rating")
)

# Join with game names
filtered_final_recommendations_df = filtered_recommendations_flat_df.join(
    game_mapping_df, 
    on="game_index", 
    how="left"
)

# Display recommendations
display(filtered_final_recommendations_df.orderBy("user_id_index", "predicted_rating", ascending=[True, False]))

# COMMAND ----------

# MAGIC %md
# MAGIC **This final cell proves that the system is not just accurate (low RMSE) but also actionable – capable of powering real-world recommendations.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 Model Comparison
# MAGIC This section compares the two trained models, the ALS model trained on the full dataset versus the one trained on the filtered dataset—based on their key performance metrics: RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).

# COMMAND ----------

# MAGIC %md
# MAGIC The code prints out a comparison table and compares test set size, user count, and game count.

# COMMAND ----------

##Compare metrics between full and filtered models
# Use metrics from the MLflow-loaded best models (full dataset and filtered dataset)
print("Model Performance Comparison:")
print(f"{'Metric':<20} {'Full Dataset Model':<20} {'Filtered Dataset Model':<20}")
print(f"{'-'*60}")
print(f"{'RMSE':<20} {rmse:<20.4f} {filtered_rmse:<20.4f}")
print(f"{'MAE':<20} {mae:<20.4f} {filtered_mae:<20.4f}")
print(f"{'Dataset Size':<20} {test_df.count():<20} {filtered_test_df.count():<20}")
print(f"{'User Count':<20} {test_df.select('user_id_index').distinct().count():<20} {filtered_test_df.select('user_id_index').distinct().count():<20}")
print(f"{'Game Count':<20} {test_df.select('game_index').distinct().count():<20} {filtered_test_df.select('game_index').distinct().count():<20}")


# COMMAND ----------

# MAGIC %md
# MAGIC **Filtered model performs slightly better with lower RMSE and MAE. Lower values indicate that the filtered model makes more accurate predictions on average, despite having access to less training data.**
# MAGIC
# MAGIC The filtered model is trained on fewer interactions and users, yet it achieves better predictive accuracy. This supports the core intuition behind filtering inactive users to train a cleaner, more focused collaborative model.

# COMMAND ----------

# MAGIC %md
# MAGIC **Below**, a grouped bar chart visually compares RMSE and MAE:
# MAGIC
# MAGIC **Blue bars = Full model; Orange bars = Filtered model**

# COMMAND ----------

# Comparison chart
metrics = ['RMSE', 'MAE']
values_full = [rmse, mae]
values_filtered = [filtered_rmse, filtered_mae]

plt.figure(figsize=(10, 6))
indices = range(len(metrics))
bar_width = 0.35
plt.bar([i - bar_width/2 for i in indices], values_full, bar_width, label='Full Model', alpha=0.7)
plt.bar([i + bar_width/2 for i in indices], values_filtered, bar_width, label='Filtered Model', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Score (lower is better)')
plt.title('Full vs Filtered Model Performance Comparison')
plt.xticks(indices, metrics)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The plot visually confirms that the filtered ALS model is more effective in minimizing prediction errors.
# MAGIC
# MAGIC Even with reduced data, the quality of interactions (active users) leads to better recommendations.

# COMMAND ----------

# MAGIC %md
# MAGIC **This comparison supports the idea that:**
# MAGIC
# MAGIC ****More data is not always better and more relevant data could be a better choice****