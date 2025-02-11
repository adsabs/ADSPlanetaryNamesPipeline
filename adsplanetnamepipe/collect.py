from typing import List, Tuple

from adsplanetnamepipe.models import KnowledgeBase, KnowledgeBaseHistory
from adsplanetnamepipe.utils.common import EntityArgs, Synonyms
from adsplanetnamepipe.utils.search_retrieval import SearchRetrieval
from adsplanetnamepipe.utils.match_excerpt import MatchExcerpt
from adsplanetnamepipe.utils.extract_keywords import ExtractKeywords
from adsplanetnamepipe.utils.adsabs_ner import ADSabsNER
from adsplanetnamepipe.utils.local_llm import LocalLLM
from adsplanetnamepipe.utils.paper_relevance import PaperRelevance


class CollectKnowldegeBase():

    """
    a class that implements a pipeline for collecting knowledge base information from scientific literature

    this class integrates various components such as search retrieval, excerpt matching,
    named entity recognition, keyword extraction, paper relevance scoring, and local
    language model processing to collect both positive and negative knowledge base entries
    """

    def __init__(self, args: EntityArgs):
        """
        initialize the CollectKnowldegeBase class

        :param args: configuration arguments for the pipeline
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
        # step 5b of the pipeline, only scoring the positive records
        self.paper_relevance = PaperRelevance(args)
        # step 5c of the pipeline, only scoring the positive records
        self.local_llm = LocalLLM(args)
        self.vocabulary = Synonyms().add_synonyms([args.target,
                                                   args.feature_type, args.feature_type_plural,
                                                   args.feature_name])

    def get_paper_relevance_score(self, doc: dict) -> float:
        """
        calculate the paper relevance score for a given document

        :param doc: dictionary containing document information
        :return: float representing the calculated paper relevance score
        """
        text = ' '.join(doc.get('title', '')) + ' ' + doc.get('abstract', '') + ' ' + doc.get('body', '')
        return self.paper_relevance.forward(
            text,
            bibstem=doc.get('bibcode', '')[4:9].strip('.'),
            databases=', '.join(doc.get('database')),
            astronomy_main_journals=self.search_retrieval.astronomy_journal_filter,
            len_existing_wikidata=len(self.extract_keywords.wiki.extract_top_keywords(text))
        )

    def get_local_llm_score(self, doc: dict, excerpt: str) -> float:
        """
        calculate the local LLM score for a given document and excerpt

        :param doc: dictionary containing document information
        :param excerpt: string containing the relevant excerpt from the document
        :return: float representing the calculated local LLM score
        """
        return self.local_llm.forward(doc['title'], doc.get('abstract', None), excerpt)

    def collect_KB_positive(self) -> List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]]:
        """
        collect positive knowledge base entries from scientific literature

        :return: list of tuples containing KnowledgeBaseHistory and associated KnowledgeBase records
        """
        collected: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = []

        docs = self.search_retrieval.collect_usgs_terms_query()
        if len(docs) > 0:
            # for each run, create a KnowledgeBaseHistory record and a list of associated KnowledgeBase records
            history_record = KnowledgeBaseHistory(
                id=None, # Set to None for now, will be updated later
                feature_name_entity=self.args.feature_name,
                feature_type_entity=self.args.feature_type,
                target_entity=self.args.target,
                named_entity_label=next(d['label'] for d in self.args.name_entity_labels if d['value'] == 1)
            )
            for i, doc in enumerate(docs):
                # for each doc temp list of KnowledgeBase records and the corresponding llm scores
                # return the collected data only if the it passes the two scores (llm and paper)
                collected_doc: List[KnowledgeBase] = []
                local_llm_scores_doc = []

                _, excerpts = self.match_excerpt.forward(doc, self.adsabs_ner)
                if excerpts:
                    paper_relevance_score = self.get_paper_relevance_score(doc)

                    # before extracting keywords from each excerpt, extract keywords from the fulltext
                    # note that for this case we are not inserting anything for excerpt, and the item_id is 0
                    tfidf_keywords = self.extract_keywords.forward_doc(doc, self.vocabulary, usgs_term=True)
                    if tfidf_keywords:
                        collected_doc.append(KnowledgeBase(
                            history_id=None,  # Set to None for now, will be updated later
                            bibcode=doc['bibcode'],
                            database=doc['database'],
                            excerpt=None,
                            keywords_item_id=0,
                            keywords=tfidf_keywords,
                            special_keywords=[],
                        ))

                    # now get keywords for each excerpt, count only if got them
                    item_id = 1
                    for excerpt in excerpts:
                        excerpt_keywords = self.extract_keywords.forward(excerpt, num_keywords=10)
                        if excerpt_keywords:
                            special_keywords = self.extract_keywords.forward_special(excerpt)
                            # include it if least one STI-keyword was selected
                            # TODO: once nasa package is installed changed replace >= to >
                            if len(special_keywords) >= 0:
                                collected_doc.append(KnowledgeBase(
                                    history_id=None,  # Set to None for now, will be updated later
                                    bibcode=doc['bibcode'],
                                    database=doc['database'],
                                    excerpt=excerpt,
                                    keywords_item_id=item_id,
                                    keywords=excerpt_keywords,
                                    special_keywords=special_keywords,
                                ))
                                local_llm_scores_doc.append(self.get_local_llm_score(doc, excerpt))
                                item_id += 1

                    # decide to add these records to the knowledge base or not
                    if item_id > 1:
                        avg_local_llm_scores = sum(local_llm_scores_doc) / len(local_llm_scores_doc)
                        # include the record for knowledge graph if
                        # 1- average of llm scores are high (experimented and 0.5 is a good threshold)
                        # 2- paper relevance score is high (experimented and 0.6 is a good threshold)
                        # 3- at least half and the original excerpts remains and
                        #    was not eliminated by the two local llm and paper relevance scores
                        if avg_local_llm_scores >= 0.5 and paper_relevance_score >= 0.6 and len(collected_doc) >= (len(excerpts) / 2):
                            # Append the tuple of history_record and collected_doc to collected
                            collected.append((history_record, collected_doc))

        return collected

    def collect_KB_negative(self) -> List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]]:
        """
        collect negative knowledge base entries from scientific literature

        :return: list of tuples containing KnowledgeBaseHistory and associated KnowledgeBase records
        """
        collected: List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]] = []

        docs = self.search_retrieval.collect_non_usgs_terms_query()
        if len(docs) > 0:
            history_record = KnowledgeBaseHistory(
                id=None, # Set to None for now, will be updated later
                feature_name_entity=self.args.feature_name,
                feature_type_entity=self.args.feature_type,
                target_entity=self.args.target,
                named_entity_label=next(d['label'] for d in self.args.name_entity_labels if d['value'] == 0)
            )
            knowledge_base_records: List[KnowledgeBase] = []

            for doc in docs:
                proceed, _ = self.match_excerpt.forward(doc, usgs_term=False)
                if proceed:
                    # for the negative side, only getting keywords from the fulltext, no excerpt extraction
                    tfidf_keywords = self.extract_keywords.forward_doc(doc, self.vocabulary, usgs_term=False)
                    if tfidf_keywords:
                        knowledge_base_records.append(KnowledgeBase(
                            history_id=None,  # Set to None for now, will be updated later
                            bibcode=doc['bibcode'],
                            database=doc['database'],
                            excerpt=None,
                            keywords_item_id=0,
                            keywords=tfidf_keywords,
                            special_keywords=[],
                        ))

            if knowledge_base_records:
                collected.append((history_record, knowledge_base_records))

        return collected

    def collect(self) -> List[Tuple[KnowledgeBaseHistory, List[KnowledgeBase]]]:
        """
        collect both positive and negative knowledge base entries

        :return: combined list of tuples containing KnowledgeBaseHistory and associated KnowledgeBase records
        """
        KB_positive = self.collect_KB_positive()
        KB_negative = self.collect_KB_negative()
        return KB_positive + KB_negative
