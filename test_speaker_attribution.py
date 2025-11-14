"""
Test Speaker Attribution in JSON Ingestion

Demonstrates how content is now formatted with speaker names
for better graph connectivity.
"""

import json
from json_ingestion import JSONIngestionAdapter


def test_speaker_attribution():
    """Test the new speaker attribution functionality"""
    
    # Test data matching your JSON structures
    test_data = {
        "qa_format": {
            "chunks": [
                {
                    "type": "qa",
                    "questioner_name": "Harssh K. Shah",
                    "answerer_name": "Vikram Mehra",
                    "question": "What challenges or headwinds is the overall industry facing other than music apps shutting down?",
                    "answer": "We are not seeing headwinds on YouTube right now. What happened in May affected advertising across media; during that period ad spending paused temporarily."
                },
                {
                    "type": "commentary", 
                    "speaker_name": "Vikram Mehra",
                    "text": "I have already shared our guidance for future. We believe music industry specific and even the video side, we are still an underutilized, under monetized entertainment industry."
                }
            ]
        },
        
        "chunk_format": [
            {
                "chunk_id": 1,
                "content": "Revenue from Operations stood at Rs. 2,300 Mn in Q2 FY26, recording a growth of 11% QoQ basis while PBT recorded a QoQ growth of 18% rising to Rs. 601 Mn",
                "document_metadata": {
                    "source_file": "Q2_PPT.json",
                    "company": "Saregama India Limited"
                }
            }
        ]
    }
    
    adapter = JSONIngestionAdapter()
    
    print("🎭 SPEAKER ATTRIBUTION TEST")
    print("="*80)
    
    # Test Q&A format
    print("\n📋 Q&A FORMAT WITH SPEAKERS:")
    print("-"*50)
    
    qa_passages = adapter.apply_ingestion_rule(
        test_data["qa_format"], 
        adapter.predefined_rules["qa_format"]
    )
    
    for i, passage in enumerate(qa_passages):
        print(f"\nPassage {i+1}:")
        print(f"📝 Content: {passage.content}")
        print(f"🏷️  Metadata: questioner={passage.passage_metadata.get('questioner')}, answerer={passage.passage_metadata.get('answerer')}")
        print()
    
    # Test chunk format (no speakers)
    print("\n📄 CHUNK FORMAT (NO SPEAKERS):")
    print("-"*50)
    
    chunk_passages = adapter.apply_ingestion_rule(
        test_data["chunk_format"],
        adapter.predefined_rules["chunk_array"]
    )
    
    for i, passage in enumerate(chunk_passages):
        print(f"\nPassage {i+1}:")
        print(f"📝 Content: {passage.content}")
        print()
    
    # Show content length filtering
    print("\n📏 CONTENT LENGTH FILTERING (min 30 chars):")
    print("-"*50)
    
    short_content_test = {
        "chunks": [
            {
                "type": "qa",
                "questioner_name": "John",
                "answerer_name": "Jane", 
                "question": "Hi?",
                "answer": "Hello!"
            },
            {
                "type": "qa",
                "questioner_name": "Alice",
                "answerer_name": "Bob",
                "question": "What is the revenue growth rate for this quarter?", 
                "answer": "The revenue growth rate is 15% quarter-over-quarter, which exceeds our expectations."
            }
        ]
    }
    
    filtered_passages = adapter.apply_ingestion_rule(
        short_content_test,
        adapter.predefined_rules["qa_format"] 
    )
    
    print(f"Original items: 2")
    print(f"Filtered passages (>30 chars): {len(filtered_passages)}")
    
    for i, passage in enumerate(filtered_passages):
        print(f"\nKept passage {i+1} ({len(passage.content)} chars):")
        print(f"📝 Content: {passage.content}")


def demonstrate_ner_impact():
    """Show how speaker attribution improves NER results"""
    
    print("\n\n🧠 NER IMPACT DEMONSTRATION")
    print("="*80)
    
    print("\n📊 BEFORE (without speaker names):")
    print("-"*50)
    old_content = "Q: What was the revenue growth?\n\nA: We achieved 15% growth this quarter."
    print(f"Content: {old_content}")
    print("Expected entities: ['15%', 'quarter']")
    print("Expected concepts: ['revenue growth']")
    print("❌ Missing: Who asked? Who answered?")
    
    print("\n📊 AFTER (with speaker names):")
    print("-"*50)  
    new_content = "Harssh K. Shah asked: What was the revenue growth?\n\nVikram Mehra answered: We achieved 15% growth this quarter."
    print(f"Content: {new_content}")
    print("Expected entities: ['Harssh K. Shah', 'Vikram Mehra', '15%', 'quarter']") 
    print("Expected concepts: ['revenue growth']")
    print("✅ Now includes: Speaker entities for graph connectivity!")
    
    print("\n🔗 GRAPH CONNECTIVITY BENEFITS:")
    print("-"*50)
    print("• Person → Question relationships")
    print("• Person → Answer relationships") 
    print("• Question → Answer flow tracking")
    print("• Speaker influence analysis")
    print("• Cross-document person matching")


if __name__ == "__main__":
    test_speaker_attribution()
    demonstrate_ner_impact()