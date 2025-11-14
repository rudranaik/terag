"""
Hybrid Edge Weight Calculator for TERAG

Calculates edge weights between passages and entities/concepts using:
1. Frequency-based (TF-IDF style)
2. Position-based (where entity appears in passage) 
3. Entity type importance
4. Context strength

All rule-based - no LLM calls required.
"""

import re
import math
from typing import List, Dict, Tuple, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class EntityMention:
    """Single mention of an entity in a passage"""
    entity_text: str
    entity_type: str  # "PERSON", "ORG", "DATE", "CONCEPT", etc.
    start_pos: int
    end_pos: int
    context_window: str  # Surrounding text


class EdgeWeightCalculator:
    """
    Hybrid edge weight calculator for TERAG graphs
    
    Combines multiple signals to determine entity/concept importance in passages:
    - Frequency: How often entity appears 
    - Position: Where in passage (start/end more important)
    - Type: Entity type importance (PERSON > DATE > CONCEPT)
    - Context: Surrounding keywords that boost relevance
    """
    
    def __init__(self):
        # Entity type importance weights
        self.type_weights = {
            "PERSON": 0.95,
            "ORG": 0.90, 
            "ORGANIZATION": 0.90,
            "LOC": 0.85,
            "LOCATION": 0.85,
            "DATE": 0.80,
            "MONEY": 0.85,
            "PERCENT": 0.80,
            "CONCEPT": 0.75,
            "UNKNOWN": 0.70
        }
        
        # Context keywords that boost entity importance
        self.importance_keywords = {
            "announced", "announced", "said", "reported", "stated", "declared",
            "CEO", "president", "director", "chairman", "founder", "leader",
            "revenue", "profit", "growth", "increase", "decrease", "performance",
            "acquired", "launched", "introduced", "developed", "created",
            "strategy", "plan", "initiative", "project", "partnership"
        }
        
    def calculate_edge_weights(
        self,
        passage_text: str,
        entities: List[str],
        concepts: List[str], 
        all_passages: List[str],
        entity_types: Dict[str, str] = None
    ) -> Dict[str, float]:
        """
        Calculate edge weights for all entities/concepts in a passage
        
        Args:
            passage_text: The passage content
            entities: List of entity strings found in passage
            concepts: List of concept strings found in passage  
            all_passages: All passages in corpus (for IDF calculation)
            entity_types: Optional mapping of entity -> type
            
        Returns:
            Dict mapping entity/concept -> edge weight (0.0 to 1.0)
        """
        weights = {}
        entity_types = entity_types or {}
        
        # Find all mentions of entities/concepts in passage
        all_terms = entities + concepts
        term_mentions = self._find_term_mentions(passage_text, all_terms, entity_types)
        
        # Calculate weights for each term
        for term in all_terms:
            if term in term_mentions:
                weight = self._calculate_single_term_weight(
                    term, 
                    term_mentions[term],
                    passage_text,
                    all_passages
                )
                weights[term] = weight
            else:
                weights[term] = 0.0
                
        return weights
    
    def _find_term_mentions(
        self, 
        passage_text: str, 
        terms: List[str],
        entity_types: Dict[str, str]
    ) -> Dict[str, List[EntityMention]]:
        """Find all mentions of terms in passage with positions"""
        mentions = {}
        passage_lower = passage_text.lower()
        
        for term in terms:
            term_mentions = []
            term_lower = term.lower()
            entity_type = entity_types.get(term, "UNKNOWN")
            
            # Find all occurrences
            start = 0
            while True:
                pos = passage_lower.find(term_lower, start)
                if pos == -1:
                    break
                    
                end_pos = pos + len(term_lower)
                context = self._get_context_window(passage_text, pos, end_pos)
                
                mention = EntityMention(
                    entity_text=term,
                    entity_type=entity_type,
                    start_pos=pos,
                    end_pos=end_pos, 
                    context_window=context
                )
                term_mentions.append(mention)
                start = end_pos
                
            if term_mentions:
                mentions[term] = term_mentions
                
        return mentions
    
    def _calculate_single_term_weight(
        self,
        term: str,
        mentions: List[EntityMention], 
        passage_text: str,
        all_passages: List[str]
    ) -> float:
        """Calculate weight for a single term based on all its mentions"""
        
        # 1. Frequency component (TF-IDF style)
        tf_weight = self._calculate_tf_weight(mentions, passage_text)
        idf_weight = self._calculate_idf_weight(term, all_passages)
        frequency_score = tf_weight * idf_weight
        
        # 2. Position component (start/end of passage more important)
        position_score = self._calculate_position_weight(mentions, passage_text)
        
        # 3. Entity type importance
        entity_type = mentions[0].entity_type if mentions else "UNKNOWN"
        type_score = self.type_weights.get(entity_type, 0.7)
        
        # 4. Context strength (surrounding important keywords)
        context_score = self._calculate_context_weight(mentions)
        
        # Hybrid combination with weights
        final_weight = (
            frequency_score * 0.35 +    # TF-IDF most important
            position_score * 0.25 +     # Position matters  
            type_score * 0.20 +         # Entity type importance
            context_score * 0.20        # Context keywords
        )
        
        # Boost for multiple mentions
        if len(mentions) > 1:
            repetition_boost = min(1.2, 1.0 + (len(mentions) - 1) * 0.1)
            final_weight *= repetition_boost
            
        return min(1.0, final_weight)  # Cap at 1.0
    
    def _calculate_tf_weight(self, mentions: List[EntityMention], passage_text: str) -> float:
        """Term frequency component"""
        term_count = len(mentions)
        # Approximate word count (simple split)
        word_count = len(passage_text.split())
        
        if word_count == 0:
            return 0.0
            
        tf = term_count / word_count
        # Log normalization to prevent very frequent terms from dominating
        return min(1.0, tf * 10)  # Scale up since TF is usually very small
    
    def _calculate_idf_weight(self, term: str, all_passages: List[str]) -> float:
        """Inverse document frequency component"""
        total_docs = len(all_passages)
        if total_docs <= 1:
            return 1.0
            
        # Count passages containing this term
        docs_with_term = sum(1 for passage in all_passages 
                           if term.lower() in passage.lower())
        
        if docs_with_term == 0:
            return 1.0
            
        # IDF formula: log(N / df)
        idf = math.log(total_docs / docs_with_term)
        # Normalize to [0, 1] range  
        max_possible_idf = math.log(total_docs)
        return idf / max_possible_idf if max_possible_idf > 0 else 0.0
    
    def _calculate_position_weight(self, mentions: List[EntityMention], passage_text: str) -> float:
        """Position-based importance (start/end of passage more important)"""
        if not mentions:
            return 0.0
            
        passage_length = len(passage_text)
        if passage_length == 0:
            return 0.0
            
        position_scores = []
        for mention in mentions:
            # Relative position in passage
            relative_pos = mention.start_pos / passage_length
            
            # U-shaped curve: start and end are most important
            if relative_pos < 0.2:  # First 20%
                pos_score = 0.9
            elif relative_pos > 0.8:  # Last 20%
                pos_score = 0.9
            elif relative_pos < 0.4 or relative_pos > 0.6:  # Second/fourth quartile
                pos_score = 0.7
            else:  # Middle 20%
                pos_score = 0.5
                
            position_scores.append(pos_score)
            
        # Return max position score (best position)
        return max(position_scores)
    
    def _calculate_context_weight(self, mentions: List[EntityMention]) -> float:
        """Context strength based on surrounding important keywords"""
        if not mentions:
            return 0.0
            
        context_scores = []
        for mention in mentions:
            context_lower = mention.context_window.lower()
            
            # Count importance keywords in context
            keyword_matches = sum(1 for keyword in self.importance_keywords
                                if keyword in context_lower)
            
            # Normalize by context window size (approximate)
            context_words = len(context_lower.split())
            context_score = keyword_matches / max(1, context_words) * 5  # Scale up
            
            context_scores.append(min(1.0, context_score))
            
        # Return max context score
        return max(context_scores) if context_scores else 0.0
    
    def _get_context_window(self, text: str, start_pos: int, end_pos: int, window_size: int = 50) -> str:
        """Get surrounding context for entity mention"""
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        return text[context_start:context_end]


# Convenience functions for integration
def calculate_passage_weights(
    passage_text: str,
    entities: List[str],
    concepts: List[str],
    all_passages: List[str] = None,
    entity_types: Dict[str, str] = None
) -> Dict[str, float]:
    """
    Quick function to calculate edge weights for a passage
    
    Args:
        passage_text: The passage content
        entities: Named entities found in passage  
        concepts: Document concepts found in passage
        all_passages: All passages in corpus (for IDF). If None, uses simple frequency
        entity_types: Optional entity type mapping
        
    Returns:
        Dict of term -> edge weight
    """
    calculator = EdgeWeightCalculator()
    
    # If no corpus provided, use just this passage for IDF
    if all_passages is None:
        all_passages = [passage_text]
        
    return calculator.calculate_edge_weights(
        passage_text, entities, concepts, all_passages, entity_types
    )


if __name__ == "__main__":
    # Test the edge weight calculator
    
    # Sample passages  
    test_passages = [
        "Tim Cook announced Apple's revenue growth of 15% in Q4 2024. The CEO emphasized strong performance in California markets.",
        "Microsoft Corporation reported significant achievements in cloud computing. Satya Nadella highlighted Azure's expansion.",
        "Apple and Microsoft compete in the technology sector. Both companies focus on innovation and market growth."
    ]
    
    # Sample entities/concepts for first passage
    entities = ["Tim Cook", "Apple", "Q4 2024", "CEO", "California"]
    concepts = ["revenue growth", "performance", "markets"]
    
    # Entity types
    entity_types = {
        "Tim Cook": "PERSON",
        "Apple": "ORG", 
        "Q4 2024": "DATE",
        "CEO": "PERSON",
        "California": "LOC"
    }
    
    print("🔗 EDGE WEIGHT CALCULATION TEST")
    print("=" * 60)
    
    calculator = EdgeWeightCalculator()
    weights = calculator.calculate_edge_weights(
        test_passages[0], 
        entities, 
        concepts,
        test_passages,
        entity_types
    )
    
    print(f"\nPassage: {test_passages[0]}")
    print("\n📊 Edge Weights:")
    
    # Sort by weight descending
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    for term, weight in sorted_weights:
        entity_type = entity_types.get(term, "CONCEPT")
        print(f"  {term:<20} ({entity_type:<8}): {weight:.3f}")
        
    print(f"\n📈 Weight Distribution:")
    weights_list = list(weights.values())
    print(f"  Max weight: {max(weights_list):.3f}")
    print(f"  Min weight: {min(weights_list):.3f}")
    print(f"  Avg weight: {sum(weights_list)/len(weights_list):.3f}")
    
    # Test position sensitivity
    print("\n🎯 Position Sensitivity Test:")
    position_test = [
        "Apple announced revenue growth.",  # Apple at start
        "The company Apple reported growth.", # Apple in middle
        "Strong growth was reported by Apple."  # Apple at end
    ]
    
    for i, passage in enumerate(position_test):
        w = calculate_passage_weights(passage, ["Apple"], [], [passage])
        print(f"  Position {i+1}: Apple weight = {w['Apple']:.3f}")