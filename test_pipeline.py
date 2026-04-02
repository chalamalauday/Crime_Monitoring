"""Quick functional test for digest_pipeline."""
import sys
sys.path.insert(0, ".")
from digest_pipeline import read_pdf, filter_articles, cluster_articles, build_cluster_groups, run_pipeline

# Test raw read
arts = read_pdf("test/Articles-14_06_2025.pdf")
print(f"Articles read: {len(arts)}")
for a in arts[:3]:
    title = a["title"][:70]
    source = a["source"]
    date = a["date"]
    print(f"  Title: {title}")
    print(f"  Source: {source}  Date: {date}")
    print()

# Test filtering
filtered = filter_articles(arts)
print(f"After filtering: {len(filtered)}/{len(arts)} articles")

# Test clustering + groups
clustered = cluster_articles(filtered)
groups = build_cluster_groups(clustered)
print(f"Cluster groups: {len(groups)}")
for g in groups[:2]:
    label = g["label"][:60]
    count = len(g["articles"])
    print(f"  Cluster: {label} | articles: {count}")
    cv = g.get("competing_views")
    if cv:
        insight = cv["insight"][:80]
        print(f"  Competing: {insight}")

# Test full pipeline
print("\n--- Full pipeline test ---")
result = run_pipeline(
    files=["test/Articles-14_06_2025.pdf"],
    keywords=None,
    date=None,
    districts=None,
    output_pdf_path="data/test_output.pdf"
)
if "error" in result:
    print(f"ERROR: {result['error']}")
else:
    print(f"Success! {len(result['filtered_articles'])} articles, {len(result['cluster_groups'])} groups")
