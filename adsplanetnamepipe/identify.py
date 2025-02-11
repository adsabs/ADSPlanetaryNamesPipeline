from typing import List, Tuple

from adsputils import load_config

config = {}
config.update(load_config())

from adsplanetnamepipe.models import NamedEntity, NamedEntityHistory
from adsplanetnamepipe.utils.common import EntityArgs, Synonyms
from adsplanetnamepipe.utils.search_retrieval import SearchRetrieval
from adsplanetnamepipe.utils.match_excerpt import MatchExcerpt
from adsplanetnamepipe.utils.extract_keywords import ExtractKeywords
from adsplanetnamepipe.utils.adsabs_ner import ADSabsNER
from adsplanetnamepipe.utils.local_llm import LocalLLM
from adsplanetnamepipe.utils.paper_relevance import PaperRelevance
from adsplanetnamepipe.utils.knowledge_graph import KnowledgeGraph
from adsplanetnamepipe.utils.label_and_confidence import LabelAndConfidence

class IdentifyPlanetaryEntities():

    """
    a class that implements a pipeline for identifying planetary entities in scientific literature

    this class integrates various components such as search retrieval, excerpt matching,
    named entity recognition, keyword extraction, knowledge graph analysis, paper relevance
    scoring, local language model processing, and entity labeling and confidence scoring
    """

    def __init__(self, args: EntityArgs, keywords_positive: List[List['str']], keywords_negative: List[List['str']]):
        """
        initialize the IdentifyPlanetaryEntities class

        :param args: configuration arguments for the pipeline
        :param keywords_positive: list of lists containing positive keywords
        :param keywords_negative: list of lists containing negative keywords
        """
        self.args = args
        # step 1 of the pipeline
        self.search_retrieval = SearchRetrieval(args)
        # step 2 of the pipeline
        self.match_excerpt = MatchExcerpt(args)
        # step 3 of the pipeline
        self.adsabs_ner = ADSabsNER(args)
        # step 4 of the pipeline
        self.extract_keywords = ExtractKeywords(args)
        # step 5a of the pipeline
        self.knowledge_graph_positive = KnowledgeGraph(args, keywords_positive,[])
        self.knowledge_graph_negative = KnowledgeGraph(args, keywords_negative,[])
        # step 5b of the pipeline
        self.paper_relevance = PaperRelevance(args)
        # step 5c of the pipeline
        self.local_llm = LocalLLM(args)
        # step 6 of the pipeline
        self.label_and_confidence = LabelAndConfidence(args)
        self.vocabulary = Synonyms().add_synonyms([args.target,
                                                   args.feature_type, args.feature_type_plural,
                                                   args.feature_name])
        self.score_format = '%.{}f'.format(config['PLANETARYNAMES_PIPELINE_FORMAT_SIGNIFICANT_DIGITS'])

    def get_knowledge_graph_score(self, keywords: List[List['str']]) -> float:
        """
        calculate the knowledge graph score based on positive and negative scores

        :param keywords: list of lists containing keywords to evaluate
        :return: float representing the calculated knowledge graph score
        """
        positive_score = self.knowledge_graph_positive.forward(keywords)
        negative_score = self.knowledge_graph_negative.forward(keywords)
        score = positive_score / (positive_score + negative_score)
        return float(self.score_format % score)

    def get_paper_relevance_score(self, doc: dict) -> float:
        """
        calculate the paper relevance score for a given document

        :param doc: dictionary containing document information
        :return: float representing the calculated paper relevance score
        """
        text = ' '.join(doc.get('title', '')) + ' ' + doc.get('abstract', '') + ' ' + doc.get('body', '')
        score = self.paper_relevance.forward(
            text,
            bibstem=doc.get('bibcode', '')[4:9].strip('.'),
            databases=', '.join(doc.get('database')),
            astronomy_main_journals=self.search_retrieval.astronomy_journal_filter,
            len_existing_wikidata=len(self.extract_keywords.wiki.extract_top_keywords(text))
        )
        return float(self.score_format % score)

    def get_local_llm_score(self, doc: dict, excerpt: str) -> float:
        """
        calculate the local LLM score for a given document and excerpt

        :param doc: dictionary containing document information
        :param excerpt: string containing the relevant excerpt from the document
        :return: float representing the calculated local LLM score
        """
        return self.local_llm.forward(doc['title'], doc.get('abstract', None), excerpt)

    def identify(self) -> List[Tuple[NamedEntityHistory, List[NamedEntity]]]:
        """
        identify planetary entities using the pipeline

        :return: list of tuples containing NamedEntityHistory and associated NamedEntity records
        """
        identified: List[Tuple[NamedEntityHistory, List[NamedEntity]]] = []

        docs = self.search_retrieval.identify_terms_query()
        if len(docs) > 0:
            # for each run, create a NamedEntityHistory record and a list of associated NamedEntity records
            history_record = NamedEntityHistory(
                id=None, # Set to None for now, will be updated later
                feature_name_entity=self.args.feature_name,
                feature_type_entity=self.args.feature_type,
                target_entity=self.args.target,
            )
            for i, doc in enumerate(docs):
                # for each doc temp list of NamedEntity records and the corresponding llm and knowledge graph scores
                # identify the entity label and confidence score for the average of these scores
                identified_docs: List[NamedEntity] = []
                local_llm_scores_doc = []
                knowledge_graph_scores_doc = []

                _, excerpts = self.match_excerpt.forward(doc, self.adsabs_ner)
                if excerpts:
                    paper_relevance_score = self.get_paper_relevance_score(doc)

                    # process each excerpt
                    item_id = 1
                    for excerpt in excerpts:
                        excerpt_keywords = self.extract_keywords.forward(excerpt, num_keywords=10)
                        if excerpt_keywords:
                            knowledge_graph_score = self.get_knowledge_graph_score(excerpt_keywords)
                            local_llm_score = self.get_local_llm_score(doc, excerpt)
                            identified_docs.append(NamedEntity(
                                history_id=None,  # Set to None for now, will be updated later
                                bibcode=doc['bibcode'],
                                database=doc['database'],
                                excerpt=excerpt,
                                keywords_item_id=item_id,
                                keywords=excerpt_keywords,
                                special_keywords=self.extract_keywords.forward_special(excerpt),
                                knowledge_graph_score=None,   # Set to None for now, will be updated later
                                paper_relevance_score=paper_relevance_score,
                                local_llm_score=None,  # Set to None for now, will be updated later
                                confidence_score=None,  # Set to None for now, will be updated later
                                named_entity_label=None,  # Set to None for now, will be updated later
                            ))
                            knowledge_graph_scores_doc.append(knowledge_graph_score)
                            local_llm_scores_doc.append(local_llm_score)
                            item_id += 1

                    # get the confidence score and named entity label, update the records,
                    # and add them to the named entity
                    if item_id > 1:
                        avg_knowledge_graph_scores = float(self.score_format % (sum(knowledge_graph_scores_doc) / len(knowledge_graph_scores_doc)))
                        avg_local_llm_scores = float(self.score_format % (sum(local_llm_scores_doc) / len(local_llm_scores_doc)))
                        # give the three scores to the keras model and get back label and confidence
                        label, score = self.label_and_confidence.forward(avg_knowledge_graph_scores,
                                                                         paper_relevance_score,
                                                                         avg_local_llm_scores)
                        # add newly acquired label/score to all the identified records
                        # also update the knowledge graph score and local llm score to the aggregated ones
                        for doc in identified_docs:
                            doc.named_entity_label = label
                            doc.confidence_score = score
                            doc.knowledge_graph_score = avg_knowledge_graph_scores
                            doc.local_llm_score = avg_local_llm_scores
                        # queue for getting saved to db
                        identified.append((history_record, identified_docs))

        return identified
