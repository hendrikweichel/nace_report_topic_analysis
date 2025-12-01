from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
import tqdm
import os
import glob
import json

pdfs_path = "projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_3/PDFs"
texts_path = "projects/nace_classification/nace_report_topic_analysis/data/datasets/reports_subset_from_full_data_3/JSONs"

pdfs = glob.glob(pdfs_path + "/*.pdf")
print(pdfs)

pipeline_options = PdfPipelineOptions(
    # your current options
)
# choose GPU device
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=8, 
    device=AcceleratorDevice.CUDA 
)
pipeline_options = PdfPipelineOptions(do_table_structure=True)
#pipeline_options.table_structure_options.mode = TableFormerMode.FAST

# assume you already built pipeline_options above
doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

os.makedirs(texts_path, exist_ok=True)

for pdf in tqdm.tqdm(pdfs):
    #try:
    if True: 
        print("Converting:", pdf)

        out_path = os.path.join(
            texts_path, os.path.splitext(os.path.basename(pdf))[0] + ".json"
        )
        
        if os.path.isfile(out_path): 
            continue

        try: 
            result = doc_converter.convert(pdf)
        except RuntimeError:
            continue
        doc = result.document
        
        pages = []
        # Option A: directly export a single page to Markdown
        print(doc.num_pages())
        for i in range(doc.num_pages()):              # 0-based
            md = doc.export_to_markdown(page_no=i)  # page-wise export
            pages.append({"page": i + 1, "markdown": md})

        # If you’d rather have plain text instead of Markdown, replace the loop by:
        # for i in range(doc.num_pages):
        #     page_doc = doc.filter(page_nrs={i})
        #     txt = page_doc.export_to_text()
        #     pages.append({"page": i + 1, "text": txt})
        out = {
            "source_file": os.path.basename(pdf),
            "num_pages": doc.num_pages(),
            "pages": pages,
        }
        
        print(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
            print("stored at", out_path)

    #except RuntimeError as e:
    #    print("RuntimeError:", e)
    #except Exception as e:
    #    print("Unexpected error:", e)
