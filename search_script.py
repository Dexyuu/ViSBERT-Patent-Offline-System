import os
import re
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from lxml import etree
from transformers import AutoTokenizer
import nltk

# Ensure stopwords are available locally
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords")
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

from model import MultimodalEncoder

# ==========================================
# CONFIGURATION
# ==========================================
RAW_DATA_DIR = "./Raw_Patents"

def get_confidence_label(score):
    if score >= 0.4387:
        return "High Similarity"
    elif score >= 0.1595:
        return "Potentially Similar"
    else:
        return "Low Similarity"

def safe_parse_xml(xml_path):
    with open(xml_path, "rb") as f:
        xml_bytes = f.read()
    xml_text = xml_bytes.decode("utf-8", errors="ignore")
    replacements = {
        "&lsqb;": "[", "&rsqb;": "]", "&nbsp;": " ", "&ndash;": "-", "&mdash;": "-",
        "&hellip;": "...", "&amp;": "&", "&apos;": "'", "&quot;": '"', "&ldquo;": '"',
        "&rdquo;": '"', "&lt;": "<", "&gt;": ">", "&agr;": "α", "&bgr;": "β",
        "&ggr;": "γ", "&dgr;": "δ", "&phgr;": "φ", "&thetgr;": "θ", "&ohgr;": "ω",
        "&null;": "",
    }
    for k, v in replacements.items():
        xml_text = xml_text.replace(k, v)
    xml_text = re.sub(r"&[a-zA-Z0-9#]+;", " ", xml_text)
    xml_text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]+", " ", xml_text)
    try:
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        return etree.fromstring(xml_text.encode("utf-8"), parser)
    except Exception as e:
        print(f"[ERROR] lxml recovery parser failed: {e}")
        return None

def extract_patent_from_xml(xml_path):
    root = safe_parse_xml(xml_path)
    if root is None:
        return {"title": "[Parsing Failed]", "abstract": "[None found]", "claims": "[None found]"}
    
    title, abstract, claims = "", "", ""
    title_tags = [".//invention-title", ".//title-of-invention", ".//technical-information/title-of-invention", ".//bibliographic-data/invention-title"]
    for tag in title_tags:
        el = root.find(tag)
        if el is not None and el.text:
            title = el.text.strip()
            break
            
    abs_nodes = root.findall(".//abstract") + root.findall(".//subdoc-abstract")
    if abs_nodes:
        abs_text = [node.text.strip() for a in abs_nodes for node in a.iter() if node.text]
        abstract = re.sub(r"\s+", " ", " ".join(abs_text)).strip()
        
    claim_nodes = root.findall(".//claims") + root.findall(".//claim")
    claim_texts = [" ".join(t.strip() for t in c.itertext() if t.strip()) for c in claim_nodes]
    if claim_texts:
        claims = re.sub(r"\s+", " ", " ".join(claim_texts)).strip()
        
    return {
        "title": title or "[None found]", 
        "abstract": abstract or "[None found]", 
        "claims": claims or "[None found]"
    }

def preview_top_results(top_indices, top_scores, db_ids):
    if not os.path.exists(RAW_DATA_DIR):
        print(f"\n[!] Raw data directory '{RAW_DATA_DIR}' not found. Skipping previews.")
        return

    print("\n" + "="*50)
    print("IN-DEPTH PATENT PREVIEWS (TOP 5)")
    print("="*50)

    for rank, (score, idx) in enumerate(zip(top_scores, top_indices), 1):
        pid = db_ids[idx]
        label = get_confidence_label(score)
        
        folder_path = os.path.join(RAW_DATA_DIR, pid)
        json_path = os.path.join(folder_path, f"{pid}.json")
        png_path = os.path.join(folder_path, f"{pid}.png")

        print(f"\n--- Rank {rank}: {pid} | Score: {score:.3f} ({label}) ---")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                
                title = data.get("title", "[No title found]")
                abstract = data.get("abstract", "[No abstract found]")
                claims = data.get("claims", "[No claims found]")
                
                print(f"TITLE:    {title}")
                print(f"ABSTRACT: {abstract[:250]}..." if len(abstract) > 250 else f"ABSTRACT: {abstract}")
                print(f"CLAIMS:   {claims[:250]}..." if len(claims) > 250 else f"CLAIMS:   {claims}")
            except Exception as e:
                print(f"[!] Failed to read JSON for {pid}: {e}")
        else:
            print("[!] No JSON text data found for this patent.")

        if os.path.exists(png_path):
            try:
                img = Image.open(png_path).convert("RGB")
                print("\n>> Popping open image window... (Close the image window to proceed to the next patent)")
                plt.figure(figsize=(7, 7))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"Figure — {pid}\nScore: {score:.3f}\nTitle: {title}")
                plt.show() 
            except Exception as e:
                print(f"[!] Failed to load image for {pid}: {e}")
        else:
            print("[!] No PNG image file found for this patent.")
            
    print("\nEnd of previews.")

@torch.no_grad()
def run_deployment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print("Initializing ViSBERT-Patent...")
    print("="*50)

    print("Loading vector database...")
    db_vectors = torch.load("offline_patent_vectors.pt", map_location=device, weights_only=True)
    db_vectors = F.normalize(db_vectors, p=2, dim=1) 
    
    with open("offline_patent_ids.json", "r") as f:
        db_ids = json.load(f)

    print("Loading classification ground truth...")
    try:
        with open("ground_truth_labels.json", "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("[!] ground_truth_labels.json not found. Classification codes will not be displayed.")
        ground_truth = {}

    print("Loading model weights (Model_Weights.pt)...")
    model = MultimodalEncoder().to(device)
    model.load_state_dict(torch.load("Model_Weights.pt", map_location=device), strict=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    print("System Ready!\n")

    while True:
        print("\n--- NEW QUERY ---")
        print("1: Text Only")
        print("2: Image Only")
        print("3: Text + Image (Manual Input)")
        print("4: Patent Folder (XML + Image)")
        print("5: Quit")
        query_type = input("Enter query type (1-5): ").strip()
        
        if query_type == "5":
            print("Shutting down engine...")
            break

        text_inputs, image_tensor = None, None

        if query_type == "1":
            query_text = input("Enter text query: ")
            text_inputs = model.preprocess_text(query_text, "", "", tokenizer, stop_words).to(device)

        elif query_type == "2":
            img_path = input("Enter path to image: ").strip().replace("\"", "").replace("'", "")
            try:
                img = Image.open(img_path).convert("RGB")
                image_tensor = model.preprocess_image(img).to(device)
                print("\nShowing Query Image (Close window to continue processing...)")
                plt.imshow(img); plt.axis("off"); plt.title("Query Diagram"); plt.show()
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

        elif query_type == "3":
            query_text = input("Enter text query: ")
            img_path = input("Enter path to image: ").strip().replace("\"", "").replace("'", "")
            try:
                img = Image.open(img_path).convert("RGB")
                image_tensor = model.preprocess_image(img).to(device)
                text_inputs = model.preprocess_text(query_text, "", "", tokenizer, stop_words).to(device)
                print("\nShowing Query Image (Close window to continue processing...)")
                plt.imshow(img); plt.axis("off"); plt.title("Query Diagram"); plt.show()
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

        elif query_type == "4":
            folder_path = input("Enter path to patent folder: ").strip().replace("\"", "").replace("'", "")
            if not os.path.isdir(folder_path):
                print("[ERROR] Directory not found.")
                continue
                
            xml_file = next((f for f in os.listdir(folder_path) if f.lower().endswith(".xml")), None)
            png_file = next((f for f in os.listdir(folder_path) if f.lower().endswith(".png")), None)

            if not xml_file:
                print("[ERROR] No XML file found in the folder.")
                continue

            xml_path = os.path.join(folder_path, xml_file)
            data = extract_patent_from_xml(xml_path)
            
            print("\n===== Extracted Patent Data =====")
            print(f"Title: {data['title']}\n")
            print(f"Abstract Preview: {data['abstract'][:250]}...\n")
            print("=====================================\n")

            text_inputs = model.preprocess_text(data['title'], data['abstract'], data['claims'], tokenizer, stop_words).to(device)

            if png_file:
                png_path = os.path.join(folder_path, png_file)
                img = Image.open(png_path).convert("RGB")
                image_tensor = model.preprocess_image(img).to(device)
                print("\nShowing Extracted Figure (Close window to continue processing...)")
                plt.imshow(img); plt.axis("off"); plt.title("Extracted Patent Diagram"); plt.show()
            else:
                print("No PNG found, proceeding with Text-Only execution.")

        else:
            print("Invalid selection.")
            continue

        print("\nExecuting Search...")
        if text_inputs and image_tensor is not None:
            print("Encoding: Fused (Text + Image)")
            q_vec = model.encode_fused(text_inputs["input_ids"], text_inputs["attention_mask"], image_tensor)
        elif text_inputs:
            print("Encoding: Text-Only (as Fused)")
            q_vec = model.encode_text_as_fused(text_inputs["input_ids"], text_inputs["attention_mask"])
        elif image_tensor is not None:
            print("Encoding: Image-Only (as Fused)")
            q_vec = model.encode_image_as_fused(image_tensor)
            
        q_vec = F.normalize(q_vec, p=2, dim=1)
        sims = (q_vec @ db_vectors.T).squeeze()
        
        top_scores, top_indices = sims.topk(50)

        print("\n" + "="*50)
        print("TOP 50 RETRIEVAL RESULTS")
        print("="*50)
        
        for rank, (score_tensor, idx_tensor) in enumerate(zip(top_scores, top_indices), 1):
            score = score_tensor.item()
            pid = db_ids[idx_tensor.item()]
            label = get_confidence_label(score)
            
            # Grab all codes for the patent
            all_codes = ground_truth.get(pid, {}).get("all", [])
            
            # Slice to only show the first 5
            display_codes = all_codes[:5]
            
            # Add a visual indicator if there are more codes hidden
            if len(all_codes) > 5:
                display_codes.append("...")
            
            print(f"{rank}. {pid} | cosine={score:.3f} ({label}) | codes={display_codes}")

        preview_choice = input("\nWould you like to preview the TOP 5 patents? (y/n): ").strip().lower()
        if preview_choice == 'y':
            preview_top_results(top_indices[:5].tolist(), top_scores[:5].tolist(), db_ids)

if __name__ == "__main__":
    run_deployment()