import os
from fpdf import FPDF

def generate_pdf():
    md_path = "HealTech_Architecture_Report.md"
    pdf_path = "HealTech_Architecture_Report.pdf"
    
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    print(f"Reading {md_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # Strip dangerous characters
    text = text.encode("ascii", "ignore").decode("ascii")

    print("Generating PDF with FPDF2...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("helvetica", size=11)
    
    # Write the entire markdown as plain text block
    pdf.multi_cell(0, 6, text=text)

    pdf.output(pdf_path)
    print(f"Success! PDF generated at: {pdf_path}")

if __name__ == "__main__":
    generate_pdf()
