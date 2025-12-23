#!/usr/bin/env python3
"""
arXiv to Zotero Automation Script

This script:
1. Searches arXiv for papers related to LLMs in education
2. Downloads PDFs temporarily
3. Uploads metadata and PDFs to Zotero via the Web API
4. Generates a human-readable digest
"""

import os
import sys
import yaml
import arxiv
import requests
from datetime import datetime
from pathlib import Path
from pyzotero import zotero
import tempfile
import json


def load_config():
    """Load configuration from config.yml"""
    config_path = Path(__file__).parent / "config.yml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_zotero_client():
    """Initialize Zotero client with credentials from environment variables"""
    library_id = os.environ.get('ZOTERO_LIBRARY_ID')
    api_key = os.environ.get('ZOTERO_API_KEY')
    library_type = os.environ.get('ZOTERO_LIBRARY_TYPE', 'user')
    
    if not library_id or not api_key:
        raise ValueError("ZOTERO_LIBRARY_ID and ZOTERO_API_KEY must be set as environment variables")
    
    return zotero.Zotero(library_id, library_type, api_key)


def search_arxiv(query, max_results=10, sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending):
    """Search arXiv for papers"""
    print(f"Searching arXiv with query: {query}")
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    results = list(search.results())
    print(f"Found {len(results)} papers")
    return results


def download_pdf(paper, temp_dir):
    """Download PDF for a paper to temporary directory"""
    try:
        pdf_path = Path(temp_dir) / f"{paper.get_short_id()}.pdf"
        paper.download_pdf(dirpath=temp_dir, filename=f"{paper.get_short_id()}.pdf")
        print(f"Downloaded PDF: {paper.get_short_id()}")
        return pdf_path
    except Exception as e:
        print(f"Error downloading PDF for {paper.get_short_id()}: {e}")
        return None


def upload_to_zotero(zot, paper, pdf_path):
    """Upload paper metadata and PDF to Zotero"""
    try:
        # Create item template for journal article
        template = zot.item_template('journalArticle')
        
        # Fill in metadata
        template['title'] = paper.title
        template['abstractNote'] = paper.summary
        template['date'] = paper.published.strftime('%Y-%m-%d')
        template['url'] = paper.entry_id
        template['publicationTitle'] = 'arXiv'
        
        # Add authors
        template['creators'] = [
            {'creatorType': 'author', 'firstName': author.name.split()[-1], 'lastName': ' '.join(author.name.split()[:-1]) if len(author.name.split()) > 1 else ''}
            for author in paper.authors
        ]
        
        # Add tags for categories
        template['tags'] = [{'tag': cat} for cat in paper.categories]
        
        # Add DOI if available
        if paper.doi:
            template['DOI'] = paper.doi
        
        # Create the item in Zotero
        resp = zot.create_items([template])
        
        if resp['success']:
            item_key = resp['success']['0']
            print(f"Created Zotero item: {item_key}")
            
            # Upload PDF if available
            if pdf_path and pdf_path.exists():
                try:
                    zot.attachment_simple([str(pdf_path)], item_key)
                    print(f"Attached PDF to item: {item_key}")
                except Exception as e:
                    print(f"Error attaching PDF: {e}")
            
            return item_key
        else:
            print(f"Error creating item: {resp}")
            return None
            
    except Exception as e:
        print(f"Error uploading to Zotero: {e}")
        return None


def generate_digest(papers, results_summary):
    """Generate a human-readable digest of the search results"""
    digest_path = Path(__file__).parent / "digest.md"
    
    with open(digest_path, 'w') as f:
        f.write("# arXiv to Zotero Digest\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Papers found: {results_summary['total_papers']}\n")
        f.write(f"- Successfully uploaded to Zotero: {results_summary['uploaded']}\n")
        f.write(f"- Failed uploads: {results_summary['failed']}\n\n")
        
        f.write("## Papers\n\n")
        
        for i, paper in enumerate(papers, 1):
            f.write(f"### {i}. {paper.title}\n\n")
            f.write(f"**Authors:** {', '.join([author.name for author in paper.authors])}\n\n")
            f.write(f"**Published:** {paper.published.strftime('%Y-%m-%d')}\n\n")
            f.write(f"**arXiv ID:** {paper.get_short_id()}\n\n")
            f.write(f"**URL:** {paper.entry_id}\n\n")
            f.write(f"**Categories:** {', '.join(paper.categories)}\n\n")
            f.write(f"**Abstract:** {paper.summary}\n\n")
            
            # Check if paper was uploaded
            status = "✅ Uploaded to Zotero" if paper.get_short_id() in results_summary['uploaded_ids'] else "❌ Upload failed"
            f.write(f"**Status:** {status}\n\n")
            f.write("---\n\n")
    
    print(f"Digest generated: {digest_path}")
    return digest_path


def main():
    """Main function"""
    try:
        # Load configuration
        config = load_config()
        
        # Parse sort configuration
        sort_by_map = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
            'submittedDate': arxiv.SortCriterion.SubmittedDate
        }
        sort_order_map = {
            'ascending': arxiv.SortOrder.Ascending,
            'descending': arxiv.SortOrder.Descending
        }
        
        sort_by = sort_by_map.get(config.get('sort_by', 'submittedDate'), arxiv.SortCriterion.SubmittedDate)
        sort_order = sort_order_map.get(config.get('sort_order', 'descending'), arxiv.SortOrder.Descending)
        
        # Search arXiv
        papers = search_arxiv(
            query=config['search_query'],
            max_results=config.get('max_results', 10),
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        if not papers:
            print("No papers found")
            # Still generate an empty digest
            results_summary = {
                'total_papers': 0,
                'uploaded': 0,
                'failed': 0,
                'uploaded_ids': []
            }
            generate_digest([], results_summary)
            return
        
        # Initialize Zotero client
        zot = get_zotero_client()
        
        # Process each paper
        results_summary = {
            'total_papers': len(papers),
            'uploaded': 0,
            'failed': 0,
            'uploaded_ids': []
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for paper in papers:
                print(f"\nProcessing: {paper.title}")
                
                # Download PDF
                pdf_path = download_pdf(paper, temp_dir)
                
                # Upload to Zotero
                item_key = upload_to_zotero(zot, paper, pdf_path)
                
                if item_key:
                    results_summary['uploaded'] += 1
                    results_summary['uploaded_ids'].append(paper.get_short_id())
                else:
                    results_summary['failed'] += 1
        
        # Generate digest
        generate_digest(papers, results_summary)
        
        print("\n=== Process Complete ===")
        print(f"Total papers: {results_summary['total_papers']}")
        print(f"Successfully uploaded: {results_summary['uploaded']}")
        print(f"Failed: {results_summary['failed']}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
