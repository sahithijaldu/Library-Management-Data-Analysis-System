import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile

from google.colab import files

# UPLOAD DATASET
uploaded = files.upload()
filename = next(iter(uploaded))

df = pd.read_csv(
    filename,
    encoding="utf-8",
    keep_default_na=False
)

df.columns = df.columns.str.strip()
df = df.replace(r"^\s*$", np.nan, regex=True)

print("=" * 60)
print("LIBRARY MANAGEMENT DATA ANALYSIS SYSTEM")
print("=" * 60)

# ORIGINAL DATASET
print("\nORIGINAL DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))
display(df.head(10))

# DATA CLEANING BEFORE
print("\nDATA CLEANING - BEFORE CLEANING")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nMissing Values Column-wise")
print(df.isnull().sum())

# REMOVE DUPLICATES
df = df.drop_duplicates().copy()

# CONVERT DATE COLUMNS
df["Issue_Date"] = pd.to_datetime(
    df["Issue_Date"],
    errors="coerce",
    format="mixed"
)

df["Return_Date"] = df["Return_Date"].replace(
    ["Not Returned", "Not returned", "N/A", "NA"],
    np.nan
)

df["Return_Date"] = pd.to_datetime(
    df["Return_Date"],
    errors="coerce",
    format="mixed"
)

# FILL MISSING CATEGORICAL VALUES
categorical_columns = [
    "Transaction_ID",
    "Student_ID",
    "Student_Name",
    "Book_ID",
    "Book_Title",
    "Author",
    "Category",
    "Status"
]

for col in categorical_columns:
    if col in df.columns and df[col].isnull().any():
        mode_values = df[col].mode()

        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

# FILL MISSING ISSUE DATES
if df["Issue_Date"].isnull().any():
    valid_dates = df["Issue_Date"].dropna()

    if not valid_dates.empty:
        df["Issue_Date"] = df["Issue_Date"].fillna(
            valid_dates.median()
        )

# FILL MISSING RETURN DATES FOR RETURNED BOOKS
returned_mask = (
    df["Status"].eq("Returned")
    & df["Return_Date"].isna()
    & df["Issue_Date"].notna()
)

df.loc[returned_mask, "Return_Date"] = (
    df.loc[returned_mask, "Issue_Date"]
    + pd.Timedelta(days=7)
)

# PRESERVE NOT RETURNED STATUS
df["Return_Date"] = df["Return_Date"].astype(object)

not_returned_mask = (
    df["Status"].isin(["Issued", "Overdue"])
    & pd.isna(df["Return_Date"])
)

df.loc[not_returned_mask, "Return_Date"] = "Not Returned"

# FILL REMAINING MISSING VALUES
for col in df.columns:
    if df[col].isnull().any():

        if col == "Return_Date":
            df[col] = df[col].fillna("Not Returned")

        elif col == "Issue_Date":
            df[col] = df[col].fillna(
                pd.Timestamp("2026-01-01")
            )

        else:
            mode_values = df[col].mode()

            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

# DATA CLEANING AFTER
print("\nDATA CLEANING - AFTER CLEANING")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

display(df.head(10))

# PREPARE ISSUE DATES
issue_dates = pd.to_datetime(
    df["Issue_Date"],
    errors="coerce"
)

# BOOK ISSUE AND RETURN ANALYSIS
print("\nBOOK ISSUE / RETURN ANALYSIS")

status_counts = (
    df["Status"]
    .value_counts()
    .reindex(
        ["Returned", "Issued", "Overdue"],
        fill_value=0
    )
)

status_summary = status_counts.reset_index()
status_summary.columns = ["Status", "Count"]

total_row = pd.DataFrame({
    "Status": ["Total"],
    "Count": [len(df)]
})

status_summary = pd.concat(
    [status_summary, total_row],
    ignore_index=True
)

print(status_summary.to_string(index=False))

# MOST BORROWED BOOKS
print("\nMOST BORROWED BOOKS")

book_counts = df["Book_Title"].value_counts()
top_books = book_counts.head(6)

display(
    top_books.rename_axis("Book_Title")
    .reset_index(name="Borrowing_Count")
)

# STUDENT BORROWING STATISTICS
print("\nSTUDENT BORROWING STATISTICS")

student_counts = df["Student_ID"].value_counts().head(10)

display(
    student_counts.rename_axis("Student_ID")
    .reset_index(name="Borrowing_Count")
)

# CATEGORY-WISE ANALYSIS
print("\nCATEGORY-WISE BORROWING")

category_counts = df["Category"].value_counts()

display(
    category_counts.rename_axis("Category")
    .reset_index(name="Borrowing_Count")
)

# MONTHLY BORROWING TREND
print("\nMONTHLY BORROWING TREND")

monthly_data = pd.DataFrame({
    "Issue_Date": issue_dates
}).dropna()

monthly_data["Month"] = monthly_data["Issue_Date"].dt.to_period("M")

monthly_counts = (
    monthly_data.groupby("Month")
    .size()
    .sort_index()
)

display(
    monthly_counts.rename_axis("Month")
    .reset_index(name="Borrowing_Count")
)

# BAR GRAPH
plt.figure(figsize=(10, 5))

top_books.plot(kind="bar")

plt.title("Most Borrowed Books")
plt.xlabel("Book Title")
plt.ylabel("Number of Borrowings")
plt.xticks(rotation=40, ha="right")

plt.tight_layout()
plt.show()

# PIE CHART
plt.figure(figsize=(8, 7))

category_counts.plot(
    kind="pie",
    autopct="%1.0f%%",
    startangle=90
)

plt.title("Book Categories")
plt.ylabel("")

plt.tight_layout()
plt.show()

# LINE GRAPH
all_months = pd.period_range(
    start="2026-01",
    end="2026-12",
    freq="M"
)

monthly_counts_full = monthly_counts.reindex(
    all_months,
    fill_value=0
)

month_labels = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(11, 5))

plt.plot(
    month_labels,
    monthly_counts_full.values,
    marker="o"
)

plt.title("Monthly Borrow Trend")
plt.xlabel("")
plt.ylabel("Number of Borrowings")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# RESULTS AND KEY INSIGHTS
print("\nRESULTS AND KEY INSIGHTS")

print("Total transactions:", len(df))

if not book_counts.empty:
    print("Most borrowed book:", book_counts.index[0])
    print("Borrowing count:", int(book_counts.iloc[0]))

if not category_counts.empty:
    print("Most popular category:", category_counts.index[0])
    print("Category borrowing count:", int(category_counts.iloc[0]))

if not student_counts.empty:
    print("Most active student:", student_counts.index[0])
    print("Student transactions:", int(student_counts.iloc[0]))

if not monthly_counts.empty:
    print("Highest borrowing month:", str(monthly_counts.idxmax()))
    print("Monthly transactions:", int(monthly_counts.max()))

print("\nISSUE / RETURN SUMMARY")
print(status_summary.to_string(index=False))

# BOOK RECOMMENDATION SYSTEM
print("\n" + "=" * 60)
print("BOOK RECOMMENDATION RESULTS")
print("=" * 60)

student_book_history = (
    df.groupby("Student_ID")["Book_Title"]
    .apply(set)
    .to_dict()
)

student_category_history = (
    df.groupby("Student_ID")["Category"]
    .apply(set)
    .to_dict()
)

available_books = df[
    ["Book_Title", "Category"]
].drop_duplicates()

recommendation_results = []

for student_id in df["Student_ID"].dropna().unique():

    borrowed_books = student_book_history.get(
        student_id, set()
    )

    preferred_categories = student_category_history.get(
        student_id, set()
    )

    recommendations = available_books[
        available_books["Category"].isin(preferred_categories)
        & ~available_books["Book_Title"].isin(borrowed_books)
    ].drop_duplicates(
        subset=["Book_Title"]
    ).head(5)

    if recommendations.empty:

        popular_books = available_books[
            ~available_books["Book_Title"].isin(borrowed_books)
        ].copy()

        popular_books["Popularity"] = (
            popular_books["Book_Title"]
            .map(book_counts)
            .fillna(0)
        )

        recommendations = (
            popular_books
            .sort_values("Popularity", ascending=False)
            .head(5)
        )

    for _, book in recommendations.iterrows():
        recommendation_results.append({
            "Student_ID": student_id,
            "Recommended_Book": book["Book_Title"],
            "Category": book["Category"]
        })

recommendation_df = pd.DataFrame(
    recommendation_results,
    columns=[
        "Student_ID",
        "Recommended_Book",
        "Category"
    ]
)

# DISPLAY RECOMMENDATIONS FOR THREE STUDENTS
if not recommendation_df.empty:

    print("\nTotal Recommendations:", len(recommendation_df))

    print("\nFirst 10 Book Recommendations:")
    display(recommendation_df.head(10))

    print("\nRecommendations for Three Students")

    for i in range(3):

        student_id = input(
            f"Enter Student ID {i + 1} (Example: S141): "
        ).strip()

        student_recommendations = recommendation_df[
            recommendation_df["Student_ID"].astype(str) == student_id
        ]

        if not student_recommendations.empty:
            print(f"\nBook Recommendations for {student_id}:")
            display(student_recommendations)
        else:
            print(f"No recommendations found for {student_id}.")

else:
    print("No book recommendations available.")

# SAVE OUTPUT FILES
df.to_csv("cleaned_library_dataset.csv", index=False)

book_counts.rename_axis("Book_Title").reset_index(
    name="Borrowing_Count"
).to_csv("book_analysis.csv", index=False)

student_counts.rename_axis("Student_ID").reset_index(
    name="Borrowing_Count"
).to_csv("student_analysis.csv", index=False)

category_counts.rename_axis("Category").reset_index(
    name="Borrowing_Count"
).to_csv("category_analysis.csv", index=False)

monthly_counts.rename_axis("Month").reset_index(
    name="Borrowing_Count"
).to_csv("monthly_analysis.csv", index=False)

status_summary.to_csv(
    "issue_return_analysis.csv",
    index=False
)

recommendation_df.to_csv(
    "book_recommendation_results.csv",
    index=False
)

print("\nALL ANALYSIS FILES CREATED SUCCESSFULLY!")

# CREATE ZIP FILE WITHOUT AUTOMATIC DOWNLOAD
output_files = [
    "cleaned_library_dataset.csv",
    "book_analysis.csv",
    "student_analysis.csv",
    "category_analysis.csv",
    "monthly_analysis.csv",
    "issue_return_analysis.csv",
    "book_recommendation_results.csv"
]

with zipfile.ZipFile(
    "Library_Analysis_Outputs.zip", "w"
) as zipf:

    for file in output_files:
        zipf.write(file)

print("ZIP FILE CREATED SUCCESSFULLY!")
print("The ZIP file is saved in the Colab session.")
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")