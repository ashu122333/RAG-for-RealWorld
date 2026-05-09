import arxiv 
import time
client = arxiv.Client()

search = arxiv.Search(
    query="retrieval augmented generation",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

for paper in search.results():
    print("TITLE:", paper.title)
    print("PUBLISHED:", paper.published)
    print("PDF:", paper.pdf_url)
    print("-" * 50)

for i, paper in enumerate(client.results(search)):

    success = False

    for attempt in range(3):

        try:
            paper.download_pdf(
                dirpath="./papers",
                filename=f"paper_{i}.pdf"
            )

            print(f"Downloaded: {paper.title}")

            success = True
            break

        except Exception as e:

            print(f"Retry {attempt+1} failed")
            print(e)

            time.sleep(5)

    if not success:
        print(f"Failed permanently: {paper.title}")


    