import regex
import math
from collections import OrderedDict
import requests

from adsputils import setup_logging, load_config

logger = setup_logging('utils')
config = {}
config.update(load_config())

import spacy
spacy_model = spacy.load("en_core_web_lg")

import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from adsplanetnamepipe.utils.common import EntityArgs


class SpacyWrapper():

    model = spacy_model

    def extract_phrases(self, annotated_text, args):
        """

        :param annotated_text:
        :param args:
        :return:
        """

        def all_nouns(phrase):
            """

            :param phrase:
            :return: all the tokens are either NOUN or PROPN
            """
            all_noun_tokens = ''
            annotated = self.model(phrase)
            if len(annotated) > 1:
                for token in annotated:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        all_noun_tokens += ' %s'%token.text
            return all_noun_tokens.strip()

        # spacy phrase extraction
        phrases = [chunk.text for chunk in annotated_text.noun_chunks]

        feature_name = args.feature_name.lower()
        noun_phrases = []
        for phrase in phrases:
            if feature_name in phrase.lower() and feature_name != phrase.strip().lower():
                noun_phrase_tokens = all_nouns(phrase)
                if noun_phrase_tokens:
                    noun_phrases.append(noun_phrase_tokens)
        return noun_phrases

    def validate_feature_name_adjective(self, annotated_text, args):
        """
        checks if the feature name has appeared as an adjective in the text,
        indicating that we cannot consider it as a valid USGS term
        for example, there are Black/Green/White entities that are moon craters

        :param annotated_text:
        :param args:
        :return: True if feature name a valid usgs term
        """
        feature_name_tag = [t.pos_ for t in annotated_text if t.text == args.feature_name]
        if len(feature_name_tag) > 0:
            if feature_name_tag[0] == 'ADJ':
                return False
        return True

    def validate_feature_name_phrase(self, annotated_text, args):
        """
        checks if the feature name has appeared as part of a phrase in the text,
        suggesting that we cannot consider it as a valid USGS term

        :param annotated_text:
        :param args:
        :return:
        """
        # get phrases that contains feature name
        phrases = self.extract_phrases(annotated_text, args)
        if not phrases:
            return True

        identifiers = set(args.feature_type.lower().split(', ') + [args.target.lower()])
        # check the phrases that contain feature name is it with the identifiers, or another token
        # if it is another token, then feature name has another context
        # ie Kaiser Crater is OK, but Russell Kaiser is not
        part_of = [phrase for phrase in phrases if phrase.count(' ') > 0 and len(set(phrase.lower().split()).intersection(identifiers)) == 0]
        return len(part_of) == 0

    def validate_feature_name(self, text, args, feature_name_span, usgs_term):
        """

        :param text:
        :param args: contains feature name identification entity
        :param feature_name_span:
        :param usgs_term: True if we think this is a usgs term
        :return:
        """
        annotated_text = self.model(text)

        if usgs_term:
            if not self.validate_feature_name_adjective(annotated_text, args):
                return False

            # need to have a few tokens before and after the feature_name
            # if does not exist, quit, since we need them to decide if the feature_name
            # has been identified as part of a phrase which makes it not usgs term
            before_feature = ' '.join(text[:feature_name_span[0]].strip().split(' ')[-4:])
            after_feature = ' '.join(text[feature_name_span[1]:].strip().split(' ')[:4])
            if not (before_feature and after_feature):
                return False

            if not self.validate_feature_name_phrase(annotated_text, args):
                return False

        # if either the term in the context of usgs is valid, or
        # it is considered to be in the context of non usgs, then should be valid
        return True

    def extract_top_keywords(self, text):
        """
        return any NER tagged entity that is only alphabet and at least 3 characters

        :param text:
        :return:
        """
        annotated_text = self.model(text)

        entities = []
        for ent in annotated_text.ents:
            if ent.text.isalpha() and len(ent.text) >= 3:
                entities.append(ent.text)
        return list(set(entities))


class YakeWrapper():

    model = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.5, dedupFunc='seqm', windowsSize=1, top=20, stopwords=None, features=None)
    lemmatizer = WordNetLemmatizer()

    def extract_phrases(self, text, args):
        """

        :param text:
        :param args:
        :return:
        """
        # yake phrase extraction
        phrases = [token for token, _ in self.model.extract_keywords(text)]

        feature_name = args.feature_name.lower()
        phrases = [phrase for phrase in phrases if
                   feature_name in phrase.lower() and feature_name != phrase.strip().lower()]

        return phrases

    def validate_feature_name(self, text, args, feature_name_span, usgs_term):
        """

        :param text:
        :param args: contains feature name identification entity
        :param feature_name_span:
        :param usgs_term: True if we think this is a usgs term
        :return:
        """
        if usgs_term:
            # need to have a few tokens before and after the feature_name
            # if does not exist, quit, since we need them to decide if the feature_name
            # has been identified as part of a phrase which makes it not usgs term
            before_feature = ' '.join(text[:feature_name_span[0]].strip().split(' ')[-4:])
            after_feature = ' '.join(text[feature_name_span[1]:].strip().split(' ')[:4])
            if not (before_feature and after_feature):
                return False

            if not self.validate_feature_name_phrase(text, args):
                return False

        # if either the term in the context of usgs is valid, or
        # it is considered to be in the context of non usgs, then should be valid
        return True

    def validate_feature_name_phrase(self, text, args):
        """
        checks if the feature name has appeared as part of a phrase in the text,
        suggesting that we cannot consider it as a valid USGS term

        :param text:
        :param args:
        :return:
        """
        # get phrases that contains feature name
        phrases = self.extract_phrases(text, args)
        if not phrases:
            return True

        identifiers = set(args.feature_type.lower().split(', ') + [args.target.lower()])
        # check the phrases that contain feature name is it with the identifiers, or another token
        # if it is another token, then feature name has another context
        # ie Kaiser Crater is OK, but Russell Kaiser is not
        part_of = [phrase for phrase in phrases if phrase.count(' ') > 0 and len(set(phrase.lower().split()).intersection(identifiers)) == 0]
        return len(part_of) == 0

    def extract_top_keywords(self, text):
        """
        return any extracted keyword that is only alphabet and at least 3 characters

        :param text:
        :return:
        """
        tokens = [self.lemmatizer.lemmatize(token) for token, _ in self.model.extract_keywords(text) if token.isalpha() and len(token) >= 3]
        return list(set(tokens))


class TfidfWrapper():

    vectorizer = TfidfVectorizer()
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    custom_stops = ['this', 'that', 'these', 'those',
                    'of', 'in', 'to', 'for', 'with', 'on', 'at',
                    'a', 'an', 'the',
                    'and', 'but', 'yet', 'so', 'for', 'nor', 'or',
                    'is', 'was', 'were', 'has', 'have', 'had',
                    'very', 'also', 'just', 'being', 'over', 'own', 'yours', 'such']
    stop_words.update(custom_stops)

    def extract_top_keywords(self, doc):
        """

        :param doc:
        :return:
        """
        segments = self.get_segments(doc)
        tfidf_vectors = self.vectorizer.fit_transform(segments)
        tfidf_features = self.vectorizer.get_feature_names_out()
        top_tfidf = sorted(zip(tfidf_features, tfidf_vectors.sum(axis=0).tolist()[0]), key=lambda x: x[1], reverse=True)[:40]
        top_tfidf = [(p, s) for p, s in top_tfidf if p not in self.stop_words]

        # lemmatize and remove small entities, also make sure the entities are all alpha characters
        top_tfidf = [(self.lemmatizer.lemmatize(p), s) for p, s in top_tfidf if p.isalpha() and len(p) >= 3]

        # get the unique entities, keep the order
        unique_entities = OrderedDict.fromkeys(p for p, _ in top_tfidf)
        return list(unique_entities.keys())

    def get_segments(self, doc):
        """

        :param doc:
        :return:
        """
        title = ' '.join(doc.get('title', ''))
        abstract = doc.get('abstract', '')

        segments = []
        if title:
            segments.append(title)
        if abstract:
            segments.append(abstract)

        body = doc.get('body', '')
        # split body into words
        tokens = body.split(' ')
        # now combine to get 20 segments
        length = len(tokens)
        chunks = math.ceil(length / 19)
        body_segments = []
        for i in list(range(length))[0::chunks]:
            body_segments.append(' '.join(tokens[i:i + chunks]))
        if body_segments:
            segments += body_segments

        return segments


class WikiWrapper():

    wiki_vocab = '''
        Synchronous Meteorological Satellite|X-Ray Timing and Polarimetry Mission|1:3-resonant trans-Neptunian object|1:4-resonant trans-Neptunian object|1:5-resonant trans-Neptunian object|2:5-resonant trans-Neptunian object|2:7-resonant trans-Neptunian object|2:9-resonant trans-Neptunian object|3:4-resonant trans-Neptunian object|3:5-resonant trans-Neptunian object|3:7-resonant trans-Neptunian object|3:8-resonant trans-Neptunian object|4:5-resonant trans-Neptunian object|4:7-resonant trans-Neptunian object|4:9-resonant trans-Neptunian object|5:9-resonant trans-Neptunian object|Compton‚ÄìBelkovich Thorium Anomaly|natural satellite of a dwarf planet|variable star with rapid variations|Alpha¬≤ Canum Venaticorum variable|low amplitude Delta Scuti variable|Advanced Extremely High Frequency|European Remote-Sensing Satellite|Galactic Radiation and Background|Global Change Observation Mission|Improved TIROS Operational System|Stretched Rohini Satellite Series|Tracking and Data Relay Satellite|Detached systems with a subgiant|History of supernova observation|hypothetical astronomical object|inner planet of the Solar System|Mars Orbiter Laser Altimeter DEM|object in the outer Solar System|object of the inner Solar System|satellite internet constellation|strewn-field-producing meteorite|artificial satellite of the Sun|geostationary balloon satellite|Iridium satellite constellation|Naval Ocean Surveillance System|resonant trans-Neptunian object|shell of an astronomical object|unconfirmed astronomical object|extreme trans-Neptunian object|High Energy Transient Explorer|Lincoln Experimental Satellite|possible red giant branch star|potentially hazardous asteroid|Raduga satellite constellation|resonant Scattered Disk Object|young stellar object candidate|Blue Large-Amplitude Pulsator|composite astronomical object|EL Canum Venaticorum variable|hypothetic terrestrial planet|inferior and superior planets|liquid mirror space telescope|low-surface-brightness galaxy|Margaritifer Sinus quadrangle|planet-crossing minor planets|rotating ellipsoidal variable|RS Canum Venaticorum variable|cataclysmic binary candidate|classical Kuiper belt object|Energetic Particles Explorer|extreme emission-line galaxy|Galactic Center of Milky Way|gravitationally bound system|intermediate-mass black hole|Joint Polar Satellite System|possible cluster of galaxies|potentially hazardous object|rotating wheel space station|Sat√©lite de Coleta de Dados|strong gravitational lensing|Composante Spatiale Optique|Earth observation satellite|evaporating gaseous globule|fast blue optical transient|FK Comae Berenices variable|Future Imagery Architecture|Gamma-ray burst progenitors|gas giant with water clouds|hot, dust-obscured galaxies|massive compact halo object|R Coronae Borealis variable|radar calibration satellite|Rapidly oscillating Ap star|Trojan camp trojan asteroid|unknown astronomical object|astrophysical X-ray source|classical Cepheid variable|direct-broadcast satellite|eclipsing binary candidate|electron-capture supernova|Gamma Cassiopeiae variable|grand design spiral galaxy|gravitational microlensing|Greek camp trojan asteroid|Intermediate spiral galaxy|Mars-crossing minor planet|Meteosat Second Generation|Orbiting Solar Observatory|pair-instability supernova|Passive Inspection CubeSat|Phoenicis Lacus quadrangle|planet of the Solar System|possible group of galaxies|ultra-compact dwarf galaxy|ultraluminous X-ray source|unbarred lenticular galaxy|Uranius group of volcanoes|weak gravitational lensing|A-type main sequence star|AM Canum Venaticorum star|astronomical radio source|B-type main sequence star|blue compact dwarf galaxy|cataclysmic variable star|Charlier (Martian crater)|circumpolar constellation|compact group of galaxies|damped Lyman-alpha system|Dirichlet‚ÄìJackson Basin|Earth Resources Satellite|ER Ursae Majoris variable|F-type main-sequence star|Freundlich-Sharonov Basin|FS Canis Majoris variable|G-type main-sequence star|high throughput satellite|interplanetary dust cloud|Ismenius Lacus quadrangle|K-type main-sequence star|long-period variable star|M-type main sequence star|Mackinac Island meteorite|Mare Acidalium quadrangle|Mare Tyrrhenum quadrangle|Meteosat third generation|O-type main sequence star|Octavia E. Butler Landing|Perrotin (Martian crater)|physical binary candidate|possible globular cluster|semiregular variable star|South Pole‚ÄìAitken basin|spectroscopic binary star|Type Ib and Ic supernovae|ultra-short period planet|UX Ursae Majoris variable|X-ray astronomy satellite|Z Camelopardalis variable|Amundsen-Ganswindt Basin|Arctic Weather Satellite|barred lenticular galaxy|brightest cluster galaxy|centimetric radio source|communications satellite|Dorsa Argentea Formation|Double Periodic Variable|flocculent spiral galaxy|Gamma Cassiopeiae analog|geosynchronous satellite|inflatable space habitat|luminous infrared galaxy|Mare Australe quadrangle|Medusae Fossae Formation|millimetric radio-source|Morelos Satellite System|Radio-quiet neutron star|reconnaissance satellite|Rotating radio transient|Schwarzschild black hole|Shelter Island meteorite|Sinus Sabaeus quadrangle|symbiotic star candidate|tidally detached exomoon|TIROS Operational System|W Ursae Majoris variable|2MASS J06205584+0434449|active galactic nucleus|amateur radio satellite|astroengineering object|barred irregular galaxy|Catena Lucretius (RNII)|Depressio Hellespontica|digital elevation model|DirecTV satellite fleet|dwarf elliptical galaxy|dwarf spheroidal galaxy|galaxy group or cluster|gamma-ray constellation|geostationary satellite|Hevelian constellations|high proper-motion star|Hyperboreus Labyrinthus|interacting binary star|Laestrygon (Laestrigon)|Lambda Eridani variable|luminous blue variables|Planet transit variable|Promontorium Archerusia|Promontorium Heraclides|pulsating variable star|satellite constellation|Schiaparelli EDM lander|Schiller-Zucchius Basin|slow irregular variable|slowly pulsating B star|small Solar System body|super soft X-ray source|superluminous supernova|supermassive black hole|Syrtis Major quadrangle|Thorne‚Äì≈ªytkow object|ultramassive black hole|V1093 Herculis variable|anomalous X-ray pulsar|asteroidal achondrites|Atmosphere of the Moon|B-type supergiant star|Bench Crater meteorite|Block Island meteorite|blue globular clusters|carbonaceous chondrite|Catena Leuschner (GDL)|Central massive object|Central Molecular Zone|Chinese constellations|Crowdfunded satellites|dwarf irregular galaxy|Earth-crosser asteroid|eruptive variable star|G-type supergiant star|Gamma Doradus variable|Hadley Rille meteorite|high mass X-ray binary|high-altitude platform|horizontal branch star|InterPlanetary Network|IW Andromedae variable|K-type supergiant star|Kerr-Newman black hole|list of lunar eclipses|Lunae Palus quadrangle|Mare Boreum quadrangle|Mars Exploration Rover|Mercury-manganese star|Mons Gruithuisen Delta|Mons Gruithuisen Gamma|Nili Patera dune field|O-type supergiant star|Orbital Test Satellite|Phaethontis quadrangle|possible peculiar star|pre-main-sequence star|Promontorium Taenarium|PV Telescopii variable|quadrangle on the Moon|Rimae Sulpicius Gallus|rotating variable star|Solar eclipses on Mars|SU Ursae Majoris stars|sub-millimetric source|trans-Neptunian object|Trinidad (Mars crater)|unbarred spiral galaxy|V777 Herculis variable|VIRTUAL PARENT alf Oph|VY Sculptoris variable|Wikimedia list article|Z Andromedae variables|zodiacal constellation|cold molecular pillar|dark globular cluster|Dorsum (astrogeology)|eclipsing binary star|Energia satellite bus|geomagnetic satellite|giant molecular cloud|low mass X-ray binary|Marsden  Group comets|moon of 216 Kleopatra|near side of the Moon|Oxia Palus quadrangle|planet beyond Neptune|possible dwarf planet|primordial black hole|protoplanetary nebula|pulsating white dwarf|Red Giant Branch star|superhabitable planet|SW Sextantis variable|SX Phoenicis variable|Twenty-eight mansions|Tyrrhenus Labyrinthus|unconfirmed exoplanet|Wright Brothers Field|X-shaped radio galaxy|XX Virginis variables|Yellowknife Bay, Mars|A-type subgiant star|Alpha Cygni variable|Angustus Labyrinthus|AR Lacertae variable|artificial satellite|astrometry satellite|astrophysical plasma|B-type subgiant star|barred spiral galaxy|Beta Cephei variable|BL Herculis variable|BY Draconis variable|cis-Neptunian object|co-orbital satellite|Coulomb-Sarton Basin|Delta Scuti variable|distant minor planet|emission-line galaxy|extragalactic planet|extremely red object|F-type subgiant star|far side of the Moon|former constellation|G-type subdwarf star|G-type subgiant star|gravitational source|GW Virginis variable|Herbig‚ÄìHaro object|HW Virginis variable|interacting galaxies|K-type subdwarf star|K-type subgiant star|L-type subdwarf star|L√≥pez (Mars crater)|Lacus Perseverantiae|lunar space elevator|M-type subdwarf star|Mare Tranquillitatis|Mendel-Rydberg Basin|navigation satellite|near-Earth supernova|near-parabolic comet|O-type subgiant star|other moons of Earth|primitive achondrite|Promontorium Agassiz|Promontorium Deville|Promontorium Fresnel|Promontorium Laplace|Soft X-ray transient|stony-iron meteorite|substellar companion|Thaumasia quadrangle|Thorne–Żytkow object|ultra diffuse galaxy|unmanned spaceflight|V361 Hydrae variable|Wolf-Rayet type star|WZ Sagittae variable|young stellar object|(126619) 2002 CX154|(224559) 2005 WU178|(241097) 2007 DU112|(256124) 2006 UK337|(286239) 2001 UR193|(307982) 2004 PG115|(461363) 2000 GQ148|(472265) 2014 SR303|(503273) 2015 PN291|(506028) 2015 HO171|(511130) 2013 WV107|(523680) 2013 YJ151|(523686) 2014 DB143|(523688) 2014 DK143|(523714) 2014 KR101|(523729) 2014 OX393|(523734) 2014 QV441|(523746) 2014 UT114|(523753) 2014 WV508|(523754) 2014 WX508|(523782) 2015 BD518|(523783) 2015 BG518|(523784) 2015 BJ518|(523791) 2015 HT171|(527443) 2007 UM126|(529939) 2010 TU191|(530055) 2010 VW224|(531682) 2012 UB178|(532094) 2013 HX156|(534074) 2014 QZ441|(534251) 2014 SW223|(535020) 2014 WG509|(535985) 2015 BF515|(535991) 2015 BD519|(538495) 2016 EB195|(548141) 2010 CD270|(558094) 2014 WX535|(559176) 2015 BE518|(559177) 2015 BF518|(559181) 2015 BX518|(559239) 2015 BF568|(562274) 2015 XW379|(575713) 2011 UT410|(577578) 2013 GW136|(577627) 2013 HZ156|(578559) 2014 DC143|(579062) 2014 KF113|(579246) 2014 OX415|(581804) 2015 KN167|(582301) 2015 RM306|(607225) 1999 XU265|552385 Rochechouart|Amazonis quadrangle|Amenthes quadrangle|Antarctic meteorite|astronomical object|astrophysical maser|atmosphere of Titan|Beta Lyrae variable|BX Circini variable|Cebrenia quadrangle|circumbinary planet|Coprates quadrangle|Cydonia Labyrinthus|Descartes Highlands|Deuteronilus Colles|Deuteronilus Mensae|dwarf spiral galaxy|Enstatite chondrite|Eridania quadrangle|extremal black hole|extreme helium star|Fanion 1 / Tripos 1|fictional moon base|Fra Mauro formation|geographic location|Groundwater on Mars|Hellespontus Montes|high-velocity cloud|hyperbolic asteroid|Hyperboreus (Lacus)|hypothetical planet|infrared dark cloud|interstellar medium|interstellar object|Lambda Bo√∂tis star|Lyman-alpha emitter|Memnonia quadrangle|metric Radio-source|near-Earth asteroid|North polar ice cap|Oceanus Procellarum|orbital power plant|Pillars of Creation|planetary-mass moon|population III star|Promontorium Agarum|Promontorium Kelvin|protoplanetary disk|pulsar timing array|Richese Labyrinthus|Rimae Vasco da Gama|rotating black hole|semidetached binary|soft gamma repeater|solar X-ray sources|South polar ice cap|star-forming galaxy|star-forming region|stellar association|stellar-wind bubble|SX Arietis variable|Syrtis Major Planum|Temporary satellite|UX Orionis variable|Uzboi-Landon-Morava|V Sagittae variable|Vallis Schr√∂dinger|W Virginis variable|Wolf‚ÄìRayet galaxy|YY Orionis variable|zero-drag satellite|ZZ Leporis variable|(150642) 2001 CZ31|(154783) 2004 PA44|(160427) 2005 RL43|(182933) 2002 GZ31|(183595) 2003 TG58|(183963) 2004 DJ64|(183964) 2004 DJ71|(187661) 2007 JG43|(242216) 2003 RN10|(250112) 2002 KY14|(281371) 2008 FC76|(305543) 2008 QY40|(338309) 2002 VR17|(471136) 2010 EO65|(471149) 2010 FB49|(482824) 2013 XC26|(501585) 2014 QA43|(505624) 2014 GU53|(518151) 2016 FH13|(523597) 2002 QX47|(523649) 2010 XZ78|(523652) 2011 LZ28|(523672) 2013 FJ28|(523676) 2013 UL10|(523682) 2014 CN23|(523708) 2014 JB80|(523709) 2014 JD80|(523710) 2014 JF80|(523733) 2014 PR70|(523740) 2014 TV85|(523743) 2014 TA86|(523770) 2014 XO40|(523771) 2014 XP40|(523776) 2014 YB50|(523797) 2016 NM56|(523798) 2017 CX33|(531942) 2013 CV82|(533396) 2014 GQ53|(533559) 2014 JG80|(540205) 2017 RS17|(547375) 2010 PC88|(550310) 2012 DD86|(567070) 2019 GB19|(574398) 2010 LO33|(576162) 2012 GT41|(576256) 2012 JH67|(576257) 2012 JD68|(576357) 2012 PV45|(578832) 2014 GP53|(578833) 2014 GV53|(578834) 2014 GF54|(578835) 2014 GL54|(578847) 2014 GJ65|(578991) 2014 JC80|(578992) 2014 JL80|(578993) 2014 JP80|(578997) 2014 JR92|(579077) 2014 LP28|(583751) 2016 NZ90|(584778) 2017 RG16|(585899) 2020 HM98|(606357) 2017 UV43|(614688) 2011 KN36|(87555) 2000 QB243|429P/LINEAR‚ÄìHill|592064 (2014 NZ65)|Adamas Labyrinthus|Amphitrites Patera|Apollinaris Tholus|Arcadia quadrangle|astrometric binary|atmosphere of Mars|BL Lacertae object|black neutron star|Chalcoporos Rupƒìs|charged black hole|Charybdis Scopulus|Chicxulub impactor|circumstellar disk|circumstellar dust|colliding galaxies|cool subdwarf star|Damocloid asteroid|Dark Doodad Nebula|derelict satellite|Diacria quadrangle|DW Lyncis variable|DY Persei variable|E. Mareotis Tholus|Elysium quadrangle|emission-line star|geodetic satellite|geopotential model|gravitational lens|GW Librae variable|Halley-type comets|Hyperborea Lingula|hypervelocity star|Iapygia quadrangle|IIE iron meteorite|interacting galaxy|intergalactic dust|intergalactic star|intermediate polar|interstellar cloud|interstellar comet|irregular variable|Kilkhampton crater|Kuiper belt object|Lacus Excellentiae|large quasar group|Little West Crater|Lutsk, Mars crater|Lyman-break galaxy|Mare Humboldtianum|Margaritifer Chaos|Margaritifer Sinus|Margaritifer Terra|McKendree cylinder|military satellite|millisecond pulsar|minor-planet group|Montes Spitzbergen|moon of 45 Eugenia|moon of 93 Minerva|N. Mareotis Tholus|Nilokeras Scopulus|Noachis quadrangle|Noctis Labyrinthus|non-periodic comet|ordinary chondrite|Piscinas Serpentes|Planitia Descensus|Population II star|pulsar wind nebula|quadrangle on Mars|research satellite|Rima Agatharchides|Sentinel programme|stellar black hole|stellar population|super star cluster|suspected variable|T-type brown dwarf|Teisserenc de Bort|terrestrial planet|tethered satellite|Tharsis quadrangle|Tianzhushan crater|triple star system|Tupile Labyrinthus|UV-emission source|Vulcanoid asteroid|W. Mareotis Tholus|Walla Walla Vallis|Y-type brown dwarf|Yangliuqing crater|(182222) 2000 YU1|(182223) 2000 YC2|(271044) 2003 FK6|(523785) 2015 CM3|(523790) 2015 HP9|(532660) 2013 VE2|(566442) 2018 EZ1|(567122) 2020 PM3|(567129) 2021 CD4|(585912) 2020 QK3|(585913) 2020 QQ7|(606653) 2018 RR2|(606985) 2020 OD8|(606988) 2020 RR5|(612095) 1999 OJ4|(95626) 2002 GZ32|384582 Juliasmith|A-type giant star|Acidalia Planitia|Aeolis quadrangle|Amazonis Planitia|Anders' Earthrise|anomalous Cepheid|Apollinaris Sulci|Arabia quadrangle|Argyre quadrangle|Ascraeus Chasmata|astrophysical jet|B-type giant star|Balloon satellite|Be X-ray binaries|Bermoothes Insula|binary black hole|Cantabras Serpens|captured asteroid|Casius quadrangle|Catena Dziewulski|Catena Timocharis|Chalcoporos Rupēs|co-orbital object|Deucalionis Regio|eccentric Jupiter|elliptical galaxy|Eridania Planitia|Eridania Scopulus|exoplanetary ring|extrasolar object|extrasolar planet|F-type giant star|Faint blue galaxy|G-type giant star|Galaxias Fluct≈´s|Galileo satellite|geographic region|GLONASS satellite|Goldilocks planet|Hadley‚ÄìApennine|Hellas quadrangle|Hephaestus Fossae|Hephaestus Rupƒìs|Her Desher Valles|Herbig Ae/Be star|hot subdwarf star|Hyperboreae Undae|hypothetical star|Iapygia (Iapigia)|interstellar dust|Klemperer rosette|Kordylewski cloud|Kreutz Sungrazers|Lacus Felicitatis|Lacus Solitudinis|lenticular galaxy|long-period comet|luminous red nova|M-type giant star|Magellanic Bridge|Magellanic spiral|Mare Fecunditatis|Markarian's Chain|Martian lava tube|Martian meteorite|Marysville crater|Mindolluin Montes|minor-planet moon|Mons Hadley Delta|Montes Archimedes|Montes Cordillera|moon of 87 Sylvia|natural satellite|navigational star|near-Earth object|Nilosyrtis Mensae|North Polar Basin|O-type giant star|Pallacopas Vallis|Palus Epidemiarum|Phlegethon Catena|polar-ring galaxy|Population I star|Poritrin Planitia|Promethei Lingula|Protonilus Mensae|Quasi-Hilda comet|Ramanathan crater|reflection nebula|relativistic star|Rimae Aristarchus|Rimae de Gasparis|Rimae Doppelmayer|Rimae Triesnecker|RR Lyrae variable|RV Tauri variable|Shalbatana Vallis|Sikun Labyrinthus|Sinus Asperitatis|space observatory|SS Cygni variable|Stimson Formation|Str√∂mgren sphere|substellar object|supernova remnant|T Tauri type star|Taniquetil Montes|Tartarus Scopulus|Tithoniae Catenae|Trivium Charontis|type Ia supernova|Type II supernova|Ulricehamn crater|Ultra-hot Jupiter|Upper Plains Unit|Vallis Schr√∂teri|Vastitas Borealis|weather satellite|Wolf‚ÄìRayet star|Woytchugga Lacuna|yellow hypergiant|yellow supergiant|Zhouzhuang crater|(90265) 2003 CL5|438P/Christensen|Advanced TIROS-N|Al-Qahira Vallis|Allegheny Vallis|Apollinaris Mons|Arcadia Planitia|Arrakis Planitia|artificial world|Asteroid capture|Athabasca Valles|Aur√©ole 1 and 2|Australe Lingula|Australe Scopuli|Boreales Scopuli|Bradbury Landing|Buzzell Planitia|Caladan Planitia|Catena Artamonov|Catena Kurchatov|Catena Mendeleev|Catena Michelson|Catena Sylvester|Catena Taruntius|cepheid variable|Ceraunius Catena|Ceraunius Fossae|Ceraunius Tholus|Chthonian planet|circumpolar star|Coronae Scopulus|Crotone (crater)|Dandelion Crater|Dark matter halo|Detached systems|Dong Fang Hong 2|Dorsa Aldrovandi|Dorsum Von Cotta|Ecaz Labyrinthus|Echoriath Montes|electroweak star|Elivagar Flumina|Elysium Planitia|Encke-type comet|Erythraeum Chaos|Esnault-Pelterie|Eumenides Dorsum|Euphrates Patera|EX Lupi variable|exozodiacal dust|Fallen Astronaut|fast radio burst|Galaxias Fluctūs|gamma-ray source|Genetaska Macula|glaciers on Mars|globular cluster|Gruithuisen city|Hagal dune field|Hanny's Voorwerp|Harmakhis Vallis|Heat Shield Rock|Helium-weak star|Hephaestus Rupēs|HI (21cm) source|Huo Hsing Vallis|Hydraotes Colles|hyperbolic comet|irregular galaxy|Jaisalmer Crater|jellyfish galaxy|Kamerlingh Onnes|Kerguelen Facula|Krocylea Insulae|Labeatis Catenae|Lacus Oblivionis|Lunar north pole|Lunar south pole|Lyman-alpha blob|Mare Moscoviense|Mare Serenitatis|Markarian galaxy|Matrioshka brain|Meridiani Planum|micro black hole|Montes Apenninus|Montes Harbinger|Montes Pyrenaeus|Montes Teneriffe|Murray Formation|Nepenthes Mensae|Nepenthes Planum|Nilokeras Mensae|O'Neill cylinder|Oenotria Scopuli|Palus Putredinis|Planctae Insulae|planetary nebula|planetary system|Promethei Chasma|Promethei Planum|pure-disc galaxy|Racetrack Lacuna|Rima Artsimovich|Rima Sheepshanks|Rimae Apollonius|Rimae Archimedes|Rimae Herigonius|Rimae Maupertuis|Rimae Posidonius|Rimae Theaetetus|ring mold crater|rings of Jupiter|rings of Neptune|Rupes Toscanelli|Santorini Facula|Saraswati Flumen|satellite galaxy|satellite system|Scamander Vallis|Seyfert 1 galaxy|Seyfert 2 galaxy|Sinus Concordiae|Sinus Gay-Lussac|Sionascaig Lacus|SRA semi-regular|SRB semi-regular|SRD semi-regular|starburst galaxy|Sulpicius Gallus|sungrazing comet|symbiotic binary|Tantalus Fluctus|Taurus‚ÄìLittrow|Tehachapi crater|Thaumasia Fossae|Thaumasia Planum|Three enclosures|Tithoniae Fossae|Tithonium Chasma|Tranquility Base|ultra-cool dwarf|ultrafaint dwarf|Valles Marineris|Vallis Inghirami|Vallis Palitzsch|volcanic plateau|Wenjiashi crater|white supergiant|Zephyria Fluctus|ZZ Ceti variable|Acidalia Colles|Aleksey Tolstoy|Amenthes Fossae|Amenthes Planum|Angelica crater|Antilia Faculae|Ap and Bp stars|Apollo asteroid|Argentea Planum|Argyre Planitia|Ariadnes Colles|artificial moon|Australe Montes|Baphyras Catena|Bashkaus Valles|Bazaruto Facula|bipolar outflow|Blue hypergiant|blue supergiant|Bosporos Planum|Brashear Crater|C-type asteroid|Catena Abulfeda|Catena Brigitte|Catena Humboldt|Celadon Flumina|Centauri Montes|Cerberus Fossae|Charitum Montes|Chasma Australe|Chryse Planitia|Chukhung Crater|Chusuk Planitia|Claritas Fossae|Columbia Valles|Concordia Regio|Coprates Catena|Coprates Chasma|Coprates Montes|coreless planet|Cybele asteroid|D-type asteroid|Daedalia Planum|deep-sky object|detached object|DIDO satellites|Dionysus Patera|Dittaino Valles|Dorsum Buckland|Dorsum Guettard|emission nebula|Erythraea Fossa|extinct volcano|extrasolar moon|Flensborg Sinus|FU Orionis star|G-type asteroid|Galaxias Colles|Galaxias Fossae|galaxy filament|gamma-ray burst|Granicus Vallis|Grj√≥t√° Valles|Hadriaca Patera|Hadriacus Palus|Hartwell crater|Hegemone Dorsum|Hellas Planitia|Hesperia Planum|Huallaga Vallis|Hufaidh Insulae|Hyblaeus Catena|Hyblaeus Chasma|Hyblaeus Fossae|Hydraotes Chaos|Hyperborei Cavi|IIAB meteorites|inferior planet|infrared source|Irensaga Montes|Isidis Planitia|Ismeniae Fossae|Issedon Paterae|J-type asteroid|Juventae Chasma|K-type asteroid|Kerr black hole|Kufstein Crater|L-type asteroid|Labeatis Fossae|Lacus Bonitatis|Lacus Lenitatis|Lacus Somniorum|Larry's Lookout|lunar lava tube|Lunar meteorite|M-type asteroid|main-belt comet|Mare Erythraeum|Mareotis Fossae|Margulis crater|Mars Pathfinder|McCauley Crater|Memnonia Fossae|Middle Crescent|Mindanao Facula|molecular cloud|Mons Vinogradov|Montes Agricola|Montes Carpatus|Montes Caucasus|Montes Riphaeus|moon of Jupiter|moon of Neptune|Moonbase Bishop|moons of Haumea|Mystic Mountain|Nectaris Fossae|Nectaris Montes|Nereidum Fretum|Nereidum Montes|Nicobar Faculae|Nilokeras Fossa|O-type asteroid|Ocean Satellite|Oceanidum Fossa|Okavango Valles|Olympica Fossae|Olympus Maculae|Olympus Paterae|orbital station|outflow channel|P-type asteroid|Palus Nebularum|Panchaia Rupƒìs|Pandorae Fretum|Paraskevopoulos|Pasithea Dorsum|Patapsco Vallis|peculiar galaxy|Perkunas Virgae|Petropavlovskiy|Phoenicis Lacus|Piscinas crater|planetary probe|Planum Angustum|Planum Australe|Planum Chronium|Polaznik Macula|possible galaxy|Promethei Rupes|Promethei Sinus|Promethei Terra|Q-type asteroid|quasi-satellite|R-type asteroid|Resurs F1-14F40|Rima Diophantus|Rima Flammarion|Rima Gay-Lussac|Rimae Alphonsus|Rimae Boscovich|Rimae Chacornac|Rimae Goclenius|Rimae Gutenberg|Rimae Mersenius|Rimae Sosigenes|Rimae Taruntius|rings of Saturn|rings of Uranus|Rossak Planitia|Rozhdestvenskiy|S-type asteroid|Scylla Scopulus|Selenean summit|Shiwanni Virgae|Sinus Meridiani|Sinus Pietrosul|Sinus Successus|Sithonius Lacus|small satellite|Smoky Mountains|South Pole Wall|SRC semiregular|stony meteorite|Sub-brown dwarf|subdwarf B star|Subdwarf O star|superior planet|surface feature|T-type asteroid|Tantalus Fossae|Tartarus Colles|Tartarus Montes|technetium star|Ten Bruggencate|Tishtrya Virgae|Tithonius Lacus|Toby Jug Nebula|trojan asteroid|Trumpler Crater|Type II Cepheid|Tyrrhena Fossae|Tyrrhena Patera|Utopia Planitia|V-type asteroid|Vallis Christel|Vallis Snellius|variable nebula|very red source|Vulcani Pelagus|Wenjiashi Mensa|Wenjiashi Tholi|Widmannst√§tten|X-type asteroid|Xanthe Scopulus|Zephyria Mensae|Zephyria Planum|Zephyria Tholus|Zephyrus Fossae|432P/PANSTARRS|435P/PANSTARRS|437P/PANSTARRS|440P/Kobayashi|Abalos Scopuli|accretion disc|Acheron Catena|Acheron Fossae|Acidalia Mensa|Aeolis Serpens|Aesacus Dorsum|Aganippe Fossa|albedo feature|Algol variable|Alpheus Colles|Amazonis Mensa|Amazonis Sulci|Amenthes Rupes|Apollo 1 Hills|Arimanes Rupes|Aromatum Chaos|Arsia Chasmata|Arsinoes Chaos|Artynia Catena|Ascraeus Mensa|Ascraeus Sulci|Ascuris Planum|Aspledon Undae|Astapus Colles|Atacama Lacuna|Atira asteroid|Atitl√°n Lacus|Atlantis basin|Atlantis Chaos|Auqakuh Vallis|Aurorae Planum|Ausonia Montes|Australe Mensa|Australe Sulci|Avernus Colles|Bamberg crater|Bathurst Inlet|bipolar nebula|blue straggler|Bosporos Rupes|Bralgu Insulae|Buvinda Vallis|Catena Gregory|Catena Littrow|celestial body|Cerberus Dorsa|Cerberus Palus|Cerberus Tholi|Chasma Boreale|Claritas Rupes|Columbia Hills|contact binary|Coogoon Valles|Coprates Labes|Coprates Mensa|Coracis Fossae|Coronae Montes|Coronae Planum|Cydonia Colles|de Vaucouleurs|Deltoton Sinus|diffuse nebula|Dorsa Andrusov|Dorsa Argentea|Dorsum Arduino|Dorsum Cushman|Dorsum Termier|edge-on galaxy|Elysium Catena|Elysium Chasma|Elysium Fossae|Enipeus Vallis|eyeball planet|Faramir Colles|Fortuna Fossae|Galaxias Chaos|galaxy cluster|Galilean moons|Gandalf Colles|Garotman Terra|Gemina Lingula|Gemini Scopuli|Giedi Planitia|Giordano Bruno|Gorgonum Chaos|GPS Block IIIA|GPS Block IIIF|Great Red Spot|group of stars|Guang Han Gong|Gujiang crater|Hadriacus Cavi|Hadriacus Mons|Hagal Planitia|Hecates Tholus|Hesperia Dorsa|Hiddekel Cavus|Hiddekel Rupes|Huberta family|Huxiang crater|Hyblaeus Dorsa|Hydaspis Chaos|Hypanis Vallis|iron meteorite|irregular moon|Ismenia Patera|Ismenius Cavus|Ismenius Lacus|Issedon Tholus|Jake Matijevic|Jarry-Desloges|Jupiter analog|Jupiter trojan|Juventae Dorsa|Juventae Mensa|Karesos Flumen|Kayangan Lacus|KƒÅr≈´n Valles|Kuo Shou Ching|Labeatis Mensa|Lacus Aestatis|Lacus Hiemalis|Lacus Luxuriae|Lacus Temporis|late-type star|Leilah Fluctus|Longboh crater|Lukeqin crater|Ma'adim Vallis|Mangala Valles|Mare Acidalium|Mare Cimmerium|Mare Desiderii|Mare Hadriacum|Mare Insularum|Mare Orientale|Mare Serpentis|Mare Tyrrhenum|Mastung crater|Matijevic Hill|Matrona Vallis|Medusae Fossae|Melrhir Lacuna|Memnonia Sulci|Merlock Montes|micrometeorite|micrometeoroid|Mithrim Montes|Mohini Fluctus|Mons Argenteus|Mons Herodotus|Mons Latreille|Mons Vitruvius|moon of Saturn|moon of Uranus|Morpheos Rupes|mountain chain|mountain range|Naktong Vallis|near-IR source|Neptune trojan|Neretva Vallis|Niliacus Lacus|Nimbus program|Nimloth Colles|nova candidate|nova-like star|O'Neill colony|OB association|Oceanidum Mons|Oenotria Plana|Oile√°n Ruaidh|Olympia Mensae|Olympia Planum|Olympia Rupƒìs|optical double|Optical pulsar|Orion variable|Ortygia Colles|Panchaia Rupēs|Pavonis Chasma|Pavonis Fossae|Penglai Insula|periodic comet|Phaenna Dorsum|Phlegra Montes|Pityusa Patera|planetary body|planetary ring|Polelya Macula|Port-Au-Prince|Promethei Mons|propeller moon|Pyramus Fossae|Pyrrhae Fossae|recurrent nova|Red hypergiant|red supergiant|Reynolds Layer|Rima Ariadaeus|Rima Cleomedes|Rima Cleopatra|Rima Furnerius|Rima Milichius|Rima Schr√∂ter|Rima Siegfried|Rimae Arzachel|Rimae Gassendi|Rimae Grimaldi|Rimae Hevelius|Rimae Hippalus|Rimae Maestlin|Rimae Menelaus|Rimae Palmieri|Rimae Petavius|Rimae Riccioli|Rimae Sirsalis|Rings of Earth|Rings of Pluto|Rombaken Sinus|Rubicon Valles|Rupes Mercator|Sabrina Vallis|Salkhad crater|Scandia Colles|scattered disc|secondary body|Seyfert galaxy|shield volcano|Shikoku Facula|Silinka Vallis|Sirenum Fossae|Sirenum Tholus|Sisyphi Montes|Sisyphi Planum|Sisyphi Tholus|Solander Point|Sotonera Lacus|space elevator|Stanford torus|stellar engine|stellar stream|stream channel|Sungari Vallis|Surinda Valles|symbiotic nova|Tanaica Montes|Tartarus Rupes|Terra Cimmeria|Tharsis Montes|Tharsis Tholus|Tianchuan Base|Tortola Facula|Tractus Catena|Tractus Fossae|Trevize Fretum|Tsiipiya Terra|type-cD galaxy|Tyrrhena Dorsa|Tyrrhena Terra|Tyrrhenus Mons|Ultima Lingula|Ultimi Scopuli|Ultimum Chasma|Ulysses Colles|Ulysses Fossae|Ulysses Patera|Ulysses Tholus|Uranius Dorsum|Uranius Fossae|Uranius Patera|Uranius Tholus|valley network|Vallis Bouvard|Vallis Capella|Vallis Krishna|Van Biesbroeck|Vening Meinesz|Vichada Valles|Vistula Valles|Von der Pahlen|Waikato Vallis|Warrego Valles|Xanthus Flumen|Xibaipo crater|Yaodian crater|Yaodian Fossae|Yaodian Tholus|Zephyrus Undae|Zhengji crater|350P/McNaught|579890 Mocnik|Abalos Colles|Aeolis Mensae|Aeolis Planum|Agatharchides|Al-Marrakushi|Amenthes Cavi|Amor asteroid|anemic galaxy|Angmar Montes|Apollo Patera|Arcadia Dorsa|Ascraeus Mons|Asopus Vallis|asteroid belt|Astra 19.2¬∞E|Astra 23.5¬∞E|Astra 28.2¬∞E|Astra 31.5¬∞E|Aten asteroid|Athena Patera|Aurorae Chaos|Aurorae Sinus|Ausonia Cavus|Ausonia Mensa|Avernus Dorsa|Avernus Rupes|Baetis Chasma|Baetis Labƒìs|Bahram Vallis|Barnacle Bill|barren planet|Batson crater|Bellinsgauzen|Bernal sphere|Biblis Patera|Biblis Tholus|Bimini Insula|binary pulsar|Bolsena Lacus|Brain terrain|Brazos Valles|Calydon Fossa|Candor Chasma|Candor Colles|Caralis Chaos|carbon planet|Cardiel Lacus|Catena Krafft|Catena Pierre|Catena Sumner|Chalce Montes|Chamba crater|chaos terrain|Charis Dorsum|Chronius Mons|Chrysas Mensa|Chryse Colles|Clanis Valles|Clasia Vallis|constellation|Counter-Earth|Cydnus Rupƒìs|Dacono crater|desert planet|Doanus Vallis|Dobrovol'skiy|Dolmed Montes|Dorsa Smirnov|Dorsa Tetyaev|Dorsa Whiston|Dorsum Bucher|Dorsum Cayeux|Dorsum Grabau|Dorsum Higazy|Dorsum Niggli|Dorsum Scilla|Dorsum Zirkel|Drilon Vallis|Durius Valles|Dzigai Vallis|Elaver Vallis|Electris Mons|Elston crater|Elysium Rupes|Erebus Montes|Eridania Lake|extinct comet|far-IR source|Fermi Bubbles|Freeman Lacus|Frento Vallis|Galaxius Mons|Ganesa Macula|Ganges Catena|Ganges Chasma|Gemma Frisius|GEO-KOMPSAT-2|Geryon Montes|Gordii Dorsum|Gordii Fossae|GPS satellite|Grjótá Valles|Gutian crater|Handir Colles|Hardin Fretum|Hayashi track|Hebrus Valles|HED meteorite|Hellas Chasma|Hellas Montes|Henry Fr√®res|Hermes Patera|Hermus Vallis|Himera Valles|hycean planet|Hydrae Chasma|Hypsas Vallis|I-4 satellite|IAB meteorite|Iberus Vallis|Icaria Fossae|Icaria Planum|Idaeus Fossae|impact crater|Izamal crater|Julius Caesar|Juventae Cavi|Juventae Fons|Kalseru Virga|Kaporo crater|Keeler Crater|KH-8 Gambit 3|Kilmia Crater|Kohlsch√ºtter|Koitere Lacus|Kozova crater|Krishtofovich|Kumbaru Sinus|Labeatis Mons|Lacus Autumni|Lacus Doloris|Lacus Timoris|lakes on Mars|Locras Valles|Longomontanus|Louros Valles|low-mass star|M√ºggel Lacus|M√Ωvatn Lacus|Malino crater|Mamers Valles|Mangala Fossa|Mare Australe|Mare Chronium|Mare Cognitum|Mare Frigoris|Mare Marginis|Mare Nectaris|Marikh Vallis|Mars monolith|Marvin crater|Maumee Valles|Mawrth Vallis|Medusae Sulci|megastructure|Mikumi crater|Mira variable|Mohoroviƒçiƒá|Mons Ardeshir|Mons Hansteen|Montes Haemus|Montes Secchi|Montes Taurus|moon of Pluto|Morava Valles|Mount Marilyn|multiple star|Nakuru Lacuna|Nanedi Valles|Nansen-Apollo|Nestus Valles|Nirgal Vallis|Noachis Terra|Noctis Fossae|Noctis Tholus|Nordenski√∂ld|North Complex|Octantis Cavi|Octantis Mons|Oenotria Cavi|Ofpe/WN9 star|Olympia Rupēs|Olympia Undae|Olympus Rupes|Ontario Lacus|Ophir Catenae|pair of stars|Parana Valles|Pavonis Sulci|peculiar star|Peneus Patera|Phison Patera|Phlegra Dorsa|Picosatellite|Pingle crater|Pityusa Rupes|Planum Boreum|Pont√©coulant|post-AGB star|Protva Valles|pulsar planet|Pyrrhae Chaos|Pyrrhae Regio|Radio Sputnik|Rahway Valles|Ravius Valles|Regiomontanus|Rhabon Valles|Rima Agricola|Rima Archytas|Rima Calippus|Rima Cardanus|Rima G√§rtner|Rima Galilaei|Rima Hansteen|Rima Hesiodus|Rima Marcello|Rima Oppolzer|Rima R√©aumur|Rima Sung-Mei|Rima T. Mayer|Rima Vladimir|Rimae Daniell|Rimae Fresnel|Rimae Hypatia|Rimae Janssen|Rimae Littrow|Rimae Maclear|Rimae Pitatus|Rimae Plinius|Rimae Ramsden|Rimae Repsold|Roemer crater|Romo Planitia|Royllo Insula|Samara Valles|Santos-Dumont|Scandia Tholi|Schwarzschild|Sedona crater|Seldon Fretum|shepherd moon|Shorty Crater|Simois Colles|Sinus Aestuum|Sinus Honoris|Sinus Lunicus|Sinus Sabaeus|Skelton Sinus|Sleepy Hollow|solar vehicle|space habitat|space station|Sparrow Lacus|Spencer Jones|spiral galaxy|Stygis Catena|Stygis Fossae|subdwarf star|super-Jupiter|Super-Neptune|Surius Vallis|Sylvia family|Tanais Fossae|Termes Vallis|Terra Sirenum|Teviot Vallis|Thyles Montes|Tinjar Valles|Tlaloc Virgae|Topola crater|Tractus Albus|Trebia Valles|trojan planet|Tsu Chung-Chi|Uranus trojan|Utopia Rupƒìs|V√§nern Lacus|Vallis Planck|Vallis Rheita|Van de Graaff|Van den Bergh|Van der Waals|variable star|Vasco da Gama|Veliko Lacuna|visual binary|Voskresenskiy|Waikare Lacus|water on Mars|Winia Fluctus|Wright Crater|Wuxing crater|Wynn-Williams|X-ray burster|Xanthe Montes|Yalaing Terra|Yelapa crater|Zenit-4–ú–ö–ú|434P/Tenagra|436P/Garradd|562964 Hudin|Abalos Mensa|Abalos Undae|Aeolis Chaos|Aeolis Dorsa|Aeolis Palus|Al-Khwarizmi|Albano Lacus|Albor Fossae|Albor Tholus|Alofi crater|Anseris Mons|Aonia Planum|Aonia Tholus|Aonium Sinus|Apsus Vallis|Arabia Terra|Arena Colles|Arena Dorsum|Argyre Rupes|Arkhangelsky|Arnus Vallis|Arwen Colles|Aureum Chaos|Avernus Cavi|Axius Valles|Bacab Virgae|Baetis Chaos|Baetis Labēs|Baetis Mensa|Banachiewicz|Banes crater|barium stars|Bayta Fretum|Belopol'skiy|Belva crater|Bennett Hill|Bilbo Colles|biosatellite|Bohnenberger|Boreas Undae|Boreum Cavus|Boussingault|Bowen-Apollo|bright giant|Buyan Insula|Candor Chaos|Candor Labes|Candor Mensa|Candor Sulci|Capri Chasma|Cavi Angusti|Cayuga Lacus|CB chondrite|Chalce Fossa|Chico Valles|Chincoteague|Chryse Chaos|CI chondrite|CK chondrite|Cleia Dorsum|Clota Vallis|Coats Facula|cold Jupiter|Coloe Fossae|compact star|Copais Palus|CR chondrite|crater chain|Crete Facula|Crveno Lacus|Cusus Valles|Cyane Catena|Cyane Fossae|Cydnus Rupēs|Dechu Crater|Deuteronilus|Dorsa Argand|Dorsa Barlow|Dorsa Brevia|Dorsa Burnet|Dorsa Geikie|Dorsa Harker|Dorsa Lister|Dorsa Mawson|Dorsa Stille|Dorsum Azara|Dorsum Cloos|Dorsum Nicol|Dorsum Oppel|Dorsum Thera|Drava Valles|Du Martheray|Dubis Vallis|Dulce Vallis|dwarf galaxy|dwarf planet|Earth analog|Earth trojan|Echus Chasma|Echus Fossae|Echus Montes|Ecumenopolis|Elpis Macula|Elysium Mons|Eratosthenes|Euripus Mons|Evening Star|Evros Vallis|Face on Mars|Farah Vallis|Field galaxy|Fracastorius|G-type stars|galaxy group|Ganges Cavus|Ganges Chaos|Ganges Mensa|Gediz Vallis|Gerasimovich|giant planet|Gigas Fossae|Green Valley|Grissom Hill|Halex Fossae|Hammar Lacus|Havel Vallis|Hebes Chasma|Hellas Chaos|Hellespontus|Henyey track|Hetpet Regio|Hibes Montes|Horarum Mons|Hs≈´anch'eng|Hubur Flumen|Husband Hill|Hydrae Cavus|Hydrae Chaos|hyperon star|Iamuna Chaos|Iamuna Dorsa|Icaria Rupes|Indus Vallis|Intelsat IVA|Iridium NEXT|Isara Valles|Isidis Dorsa|Ituxi Vallis|Izola crater|Jerid Lacuna|Jingpo Lacus|Johannesburg|Jovis Fossae|Jovis Tholus|Jun√≠n Lacus|Kārūn Valles|Kasei Valles|Kasei Vallis|Kenge Crater|KH-10 Dorian|KH-11 KENNEN|KH-6 Lanyard|KH-9 Hexagon|Konstantinov|Kovalevskaya|Kutch Lacuna|La Condamine|Labou Vallis|Lacus Gaudii|Lacus Mortis|Ladoga Lacus|Ladon Valles|Lethe Vallis|Libya Montes|Licus Vallis|Liris Valles|LL chondrite|Lobachevskiy|Logtak Lacus|Loire Valles|Lucus Planum|Lunae Planum|lunar crater|Mackay Lacus|Malea Patera|Malea Planum|Mandel'shtam|Mare Crisium|Mare Humorum|Mare Imbrium|Mare Ingenii|Mare Sirenum|Mare Smythii|Mare Spumans|Mare Undarum|Mare Vaporum|Marius Hills|Marte Vallis|Mayda Insula|Melas Chasma|Melas Fossae|Meroe Patera|Meshcherskiy|mesosiderite|Mini-Neptune|Minio Vallis|minor planet|missing mass|Misty Montes|Moeris Lacus|Mons Amp√®re|Mons Argaeus|Mons Bradley|Mons Delisle|Mons Huygens|Mons La Hire|Mons Maraldi|Mons R√ºmker|Montes Alpes|Montes Recti|moon of Mars|Moria Montes|Morning Star|moving group|Munda Vallis|NASA program|Navua Valles|neutron star|New Plymouth|Ngami Lacuna|Nicer Vallis|Nicoya Sinus|Niger Vallis|Nilus Mensae|Nix Olympica|Nordenskiöld|ocean planet|Ochus Valles|Ogygis Regio|Ogygis Rupes|Ogygis Undae|Oituz crater|Oltis Valles|Olympia Cavi|Olympus Mons|Omar Khayyam|Oneida Lacus|open cluster|Ophir Chasma|Ophir Planum|orange giant|orbital ring|Orcus Patera|Orson Welles|Osuga Valles|OU Geminorum|outer planet|Padus Vallis|Parva Planum|Pavonis Mons|Peace Vallis|Peneus Palus|Peraea Cavus|PG 1159 star|Philadelphia|Phison Rupes|Phrixi Regio|Phrixi Rupes|Piazzi Smyth|planetesimal|Pliva Vallis|Project Echo|promontorium|Protei Regio|puffy planet|radio galaxy|regular moon|Reiner Gamma|Rerir Montes|Reull Vallis|Rima Bradley|Rima Brayley|Rima Delisle|Rima G. Bond|Rima Hyginus|Rima Krieger|Rima Messier|Rima Yangel'|Rimae Darwin|Rimae Gerard|Rimae Pettit|Rimae R√∂mer|Rimae Ritter|Rimae Secchi|Robert Sharp|rogue planet|Rohe Fluctus|runaway star|Rupes Cauchy|Rupes Kelvin|Rupes Liebig|Rupes Tenuis|Sabis Vallis|Sacra Fossae|Scandia Cavi|Schiaparelli|Schr√∂dinger|Senus Vallis|Sepik Vallis|Sharp-Apollo|Silberschlag|Siloe Patera|Simud Valles|Sinai Fossae|Sinai Planum|Sinus Amoris|Sinus Iridum|Sirenum Mons|Sisyphi Cavi|Smoluchowski|solar analog|Solis Planum|Sotra Patera|space debris|star cluster|Steno-Apollo|Strange star|Stura Vallis|subsatellite|Subur Vallis|Sulci Gordii|supercluster|supervolcano|Suzhi Crater|Syria Colles|Syria Planum|Syrtis Major|Tader Valles|Tagus Valles|Tempe Colles|Tempe Fossae|Tenuis Cavus|Tenuis Mensa|Terra Nivium|Terra Sabaea|Texel Facula|Theon Junior|Theon Senior|Theophrastus|Thyles Rupes|Tigre Valles|Tinia Valles|Tinto Vallis|Tisia Valles|Tollan Terra|Towada Lacus|Tracy's Rock|Tsiolkovskiy|Tsomgo Lacus|Tyras Vallis|Uanui Virgae|Udzha Crater|Ulyxis Rupes|Uranius Mons|Utopia Rupēs|Uyuni Lacuna|Uzboi Vallis|Vallis Alpes|Vallis Baade|vampire star|Varus Valles|Vedra Valles|Venus trojan|Verde Vallis|Von B√©k√©sy|Von K√°rm√°n|Wakasa Sinus|Walvis Sinus|weapon model|X-ray binary|X-ray pulsar|Xanthe Chaos|Xanthe Dorsa|Xanthe Terra|Yaonis Regio|yellow giant|yellow stars|Yelwa Crater|Zarqa Valles|430P/Scotti|431P/Scotti|439P/LINEAR|A-type star|Abaya Lacus|Abus Vallis|Acapulcoite|Aeolis Mons|Alba Catena|Alba Fossae|Alba Patera|Albategnius|Alpetragius|Anaximander|Anio Valles|Aonia Terra|Ara Fluctus|Aram Dorsum|Arda Valles|Ares Vallis|Argyre Cavi|Argyre Mons|Aristarchus|Aristoteles|Arnar Sinus|Arsia Sulci|Artsimovich|Atrax Fossa|Auxo Dorsum|binary star|Bishop Ring|black dwarf|blue object|Boguslawsky|Bok globule|Boreosyrtis|Bounce Rock|Bremerhaven|brown dwarf|Buys-Ballot|C. Herschel|Capri Mensa|carbon star|Catena Davy|Catena Yuri|Ceti Chasma|Chacornac A|Chacornac B|Chacornac C|Chacornac D|Chacornac E|Chacornac F|Champollion|chassignite|Chawla Hill|Chersonesus|Chrysokeras|Cleostratus|Colles Nili|cosmic dust|Cruz crater|Cyane Sulci|Daga Vallis|dark galaxy|dark nebula|De Gasparis|de Gerlache|debris disk|Deseilligny|Deva Vallis|disc galaxy|Doppelmayer|Dorsa Ewing|Dorsa Rubey|Dorsa Sorby|Dorsum Gast|Dorsum Heim|Dorsum Owen|double star|Duke Island|Echus Chaos|Echus Palus|Eden Patera|Elba Facula|Elektro‚ÄìL|Engel'gardt|Erebor Mons|Eurus Undae|exotic star|Eyre Lacuna|Felis Dorsa|Flaugergues|former lake|Frozen star|Gabes Sinus|galaxy wall|Garu crater|Gigas Sulci|Goldschmidt|Gonnus Mons|Gram Montes|Great Comet|Gruemberger|Gruithuisen|H chondrite|H II region|H. G. Wells|H√©derv√°ri|Hebes Mensa|Helium star|Henry Moore|Hera Patera|Herculaneum|Hertzsprung|Hess-Apollo|Hexahedrite|Hinshelwood|Hippocrates|Hobal Virga|Hoffmeister|hot Jupiter|Hot Neptune|Hotei Arcus|Hotei Regio|Hrad Valles|hydrosphere|Ibn Battuta|Intelsat 27|Intelsat II|iron planet|Ister Chaos|Izola Mensa|J. Herschel|Jezero Mons|Jori crater|Jules Verne|K chondrite|K-type star|Kaliningrad|KH-7 Gambit|Kibal'chich|Kolh√∂rster|Kosmos-2510|Kraken Mare|Krusenstern|Kuiper belt|L chondrite|La P√©rouse|labyrinthus|Lacus Veris|Lanao Lacus|Landsteiner|Last Chance|lava planet|Lebedinskiy|Leeuwenhoek|Levi-Civita|Lichtenberg|Ligeia Mare|Lobo Vallis|Lunae Mensa|Lunae Palus|Lycus Sulci|M-type star|M. Anderson|Maja Valles|Maji crater|Malam Cavus|Mare Anguis|Mare Boreum|Mare Hiemis|Mare Nubium|Mare Parvum|Mars crater|Mars trojan|McCool Hill|Melas Dorsa|Melas Labes|Melas Mensa|Microquasar|Milankoviƒç|Mohammed VI|Mohe crater|Mons Andr√©|Mons Dieter|Mons Hadley|Montes Jura|Montes Rook|Montgolfier|Moon Museum|Moray Sinus|Mosa Vallis|Mutus-Vlacq|Nako crater|Napo Vallis|Naro Vallis|NATO SATCOM|Naturaliste|Neagh Lacus|Neith Regio|Nili Fossae|Nili Patera|Nili Planum|Nili Tholus|Nilo Syrtis|Nilus Chaos|Nilus Dorsa|Notus Undae|O-type star|Oahu Facula|octahedrite|Ohrid Lacus|Okahu Sinus|Ophir Cavus|Ophir Labes|Ophir Mensa|Oppenheimer|Opportunity|Orbcomm-OG2|Oxia Colles|Oxia Planum|Oxus Patera|Palus Somni|Parent body|Patos Sinus|Peraea Mons|Phaethontis|Piccolomini|Pindus Mons|Planck star|Pot of Gold|protogalaxy|protoplanet|Puget Sinus|pulsar kick|Qara Crater|Qidu crater|Qidu Fossae|R chondrite|Ravi Vallis|Reichenbach|Rima Carmen|Rima Cauchy|Rima Draper|Rima Hadley|Rima Jansen|Rima Mairan|Rima Marius|Rima Rudolf|Rima Wan-Yu|Rimae Atlas|Rimae B√ºrg|Rimae Focas|Rimae Kopff|Rimae Opelt|Rimae Parry|Rimae Plato|Rimae Prinz|Rimae Zupus|ring galaxy|ring system|Rittenhouse|Roddenberry|Rosenberger|Ross Crater|rubble pile|Runa Vallis|Rupes Altai|Rupes Boris|Rupes Recta|S-type star|Sacra Dorsa|Sacra Mensa|Sacra Sulci|Santa Maria|Sava Vallis|Schjellerup|Schlesinger|Schomberger|Sera crater|Sevan Lacus|Shakespeare|Sheepshanks|shergottite|Sherrington|Shoji Lacus|Sinai Dorsa|Sinus Fidei|Sinus Medii|Sinus Roris|Siton Undae|Solis Dorsa|Solis Lacus|space probe|Spallanzani|star system|Styx Dorsum|super-Earth|superbubble|Tana Vallis|Tarq Crater|Taurus Void|Taus Vallis|Tempe Mensa|Tempe Terra|Tianhe Base|Tikhonravov|Triesnecker|trojan moon|Trold Sinus|Tycho Brahe|Ulla family|Ulugh Beigh|Urmia Lacus|Usiku Cavus|Vallis Bohr|Van den Bos|Vid Flumina|Vinogradsky|void galaxy|von Behring|Von Neumann|Weierstrass|white dwarf|Wr√≥blewski|Wurzelbauer|X-ray stars|Yantar-1KFT|Yantar-4K2M|Yantar-4KS1|Yellowknife|Zaim crater|Zeus Patera|Zhang Yuzhe|√Öngstr√∂m|≈†afa≈ô√≠k|2014 UO224|2014 WV535|2014 WW509|2015 BH518|2015 BK518|2015 BP518|2015 BX603|2015 FK345|2015 PJ311|2015 RD277|2015 RF277|2015 RH277|2015 RV245|2015 VE164|2015 VF164|2015 VT152|2016 GC241|2016 GR206|2016 GZ251|427P/ATLAS|428P/Gibbs|8 Homeward|achondrite|Adirondack|Alfraganus|Aliacensis|Alkyonides|Anaxagoras|Anaximenes|Aonia Mons|Apollonius|Aram Chaos|Archimedes|Argelander|Aristillus|Arsia Mons|Astra 5¬∞E|atmosphere|Aura Undae|Barabashov|Beijerinck|Bel'kovich|Bergstrand|Biosputnik|Birmingham|black hole|Blanchinus|blue dwarf|blue giant|Boeddicker|Bondarenko|Bonneville|boson star|Brachinite|Bridgetown|Bronkhorst|Brown Hill|Bullialdus|Burckhardt|C. Mayer D|Cannizzaro|Cape Verde|Carmichael|Carrington|Cassegrain|Cavalerius|Censorinus|Ceti Labes|Ceti Mensa|Chamberlin|Chang Heng|Changs≈èng|Charleston|Chernyshev|Chevallier|Comas Sola|comet dust|Copernicus|Coronation|d'Alembert|D'Arsonval|Dalu Cavus|Dao Vallis|Democritus|dense core|depression|Deslandres|Diophantus|Dorsa Cato|Dorsa Dana|dwarf nova|dwarf star|Dziewulski|Earthlight|Eberswalde|Eir Macula|El Capitan|Eos Chasma|Epimenides|Evpatoriya|Fahrenheit|fast novae|Feia Lacus|Feoktistov|FitzGerald|fixed star|Flammarion|flare star|Fontenelle|FORMOSAT-3|FORMOSAT-7|Fraunhofer|Freundlich|Gay-Lussac|giant star|green star|Gullstrand|H I region|Hargreaves|Heraclitus|Herigonius|Hipparchus|Holetschek|Home Plate|Hortensius|Houtermans|hypergiant|Iani Chaos|Ibn Firnas|ice planet|II Thyle I|ionosphere|Ius Chasma|Karpinskiy|Katchalsky|KH-5 Argon|Kivu Lacus|Kondratyuk|Kostinskiy|Koval'skiy|Krasovskiy|kugelblitz|Lacus Odii|Lacus Spei|Layl Cavus|Le Monnier|Le Verrier|Lippershey|lost comet|Lubiniezky|lunar dome|lunar mare|Lunokhod 1|Lunokhod 2|Mad Vallis|Magelhaens|Mandrake 2|Marco Polo|Mare Novum|Mars rover|Maupertuis|Maurolycus|McLaughlin|mega-Earth|Mesoplanet|Mezzoramia|Milankovic|Moa Valles|Mohe Tholi|Molesworth|Molniya-1+|Molniya-1T|Molniya-3K|Mons Agnes|Mons Dilip|Mons Ganau|Mons Penck|Mons Piton|Mons Wolff|Mont Blanc|Montevallo|N√∂ggerath|Nasireddin|Nautilus-X|Nia Fossae|Nia Tholus|Nia Vallis|North Star|Novus Mons|OH/IR star|Oodnadatta|Oti Fossae|OVV quasar|Oxia Chaos|Oxia Palus|Oxus Cavus|Paracelsus|Pea galaxy|Peirescius|Perepelkin|Phi Pegasi|Phocylides|PocketQube|Portsmouth|Posidonius|preon star|Protagoras|Protonilus|Ptolemaeus|Punga Mare|Pythagoras|quadrangle|quark star|Quark-nova|Quasi-star|Rabbi Levi|radio star|Richardson|Rima Billy|Rima Conon|Rima Dawes|Rima Euler|Rima Reiko|Rima Sharp|Rima Suess|Rima Zahia|Rimae Bode|Rimae Hase|Rocknest 3|Rutherford|Rutherfurd|S√∂mmering|Sacrobosco|Santa Cruz|Sasserides|Sch√∂nfeld|Schaeberle|Schliemann|Schumacher|Sentinel-1|Sentinel-2|Sentinel-3|Sentinel-6|Shackleton|Shangri-La|shellworld|Shirakatsi|Shternberg|Shuckburgh|Siedentopf|Sierpinski|Sklodowska|slow novae|Somerville|Sommerfeld|South Star|space dock|spiral arm|St. George|Str√∂mgren|Super-puff|supergiant|Surveyor 5|Suwa Lacus|Syracuse 4|Syria Mons|Sytinskaya|Teisserenc|Tereshkova|The Helmet|Theaetetus|Theophilus|Thymiamata|Tianlian I|Tidal tail|Tikhomirov|Timiryazev|Timocharis|Timoshenko|Tiu Valles|Torricelli|Toscanelli|Tunu Sinus|Tuscaloosa|Una Vallis|V√§is√§l√§|Van Albada|Van Maanen|Van't Hoff|Vashakidze|Vendelinus|Vernadskiy|Vetchinkin|Vinogradov|Vis Facula|von Baeyer|Von Zeipel|white hole|Wilmington|Wislicenus|Wrottesley|Xenophanes|Yablochkov|Yantar-4K1|Yantar-4K2|Z√§hringer|Zhiritskiy|Zhukovskiy|2014 FM72|2014 YX49|2015 GA54|2015 GB54|2015 GY53|2015 HX10|2016 FD14|2016 QF86|2017 QF33|2017 UX51|2019 AJ16|2019 GN22|2019 UO14|2020 BF12|2021 CO10|2021 JK10|2021 PU23|Abul Wafa|Aethiopis|Al-Biruni|Alba Mons|Alexander|Alphonsus|Amsterdam|Andersson|Annapolis|Ansgarius|Antoniadi|Ariadaeus|Armi≈Ñski|Armstrong|Arrhenius|Artamonov|Aryabhata|Autolycus|B√ºsching|Bakhuysen|Bania Arm|Barringer|Becquerel|Beƒçv√°≈ô|Bernoulli|Berzelius|Bessarion|Bianchini|Birkeland|Blagovest|Blancanus|Blanchard|Bobillier|Boltzmann|Bonestell|Boot Hill|Boscovich|Bredikhin|Brianchon|Burroughs|Calahorra|Canaveral|Carpenter|Catharina|Cavendish|CEMP star|Ceraunius|Chacornac|Chang-Ngo|Changsŏng|Chaplygin|Chatturat|Chauvenet|Chebyshev|chondrite|Chr√©tien|Chupadero|Cleomedes|Cockcroft|Condorcet|Crommelin|Ctesibius|Damoiseau|Danielson|dark star|De Forest|De La Rue|De Moraes|De Morgan|De Sitter|Dellinger|Dembowski|Desargues|Descartes|Diogenite|Dionysius|Dioscuria|Dirichlet|Dokuchaev|Doom Mons|DRAGONSat|Drygalski|Dunthorne|Eddington|Eichstadt|Einthoven|Ejriksson|Emma Dean|Endeavour|Endurance|Eos Chaos|Eos Mensa|Escalante|Esclangon|Euphrates|Evdokimov|Fabricius|Fernelius|Feuill√©e|Flamsteed|Florensky|formation|Fra Mauro|Furnerius|Ganswindt|gas dwarf|gas giant|Gaudibert|Gernsback|GLONASS-K|GLONASS-M|Goclenius|Goldstone|gravastar|Grindavik|Guabonito|Guillaume|Gutenberg|Haidinger|Haiyang-2|Handlov√°|Hargraves|Heaviside|Hecataeus|Helmholtz|Henderson|Herodotus|Heyrovsky|Hildegard|Howardite|Ibn Bajja|Ibn Yunus|Ibn-Rushd|Ibragimov|ice giant|Inghirami|Innsbruck|iron star|Ius Labes|Ius Mensa|Jamestown|Johnstown|Kagoshima|Kanopus-V|Khvol'son|Kirchhoff|Kleymenov|Knox-Shaw|Koval'sky|Kurchatov|La Caille|Lallemand|Langrenus|Lauritsen|lava tube|Lavoisier|Le Gentil|Lederberg|Lema√Ætre|Leucippus|Leuschner|Lexington|Lindbergh|Liouville|Littleton|Lodranite|Lomonosov|Lucretius|Maclaurin|MacMillan|Macrobius|Maidstone|Makhambet|Maricourt|Marsquake|Maskelyne|McAuliffe|Mechnikov|megamaser|Mendeleev|Mercurius|Mersenius|meteorite|meteoroid|Mezentsev|Michelson|micronova|Milichius|Millochau|Minkowski|Mistretta|Molniya-1|Molniya-2|Molniya-3|Mons Esam|Mons Moro|Mons Pico|Mons Usov|Montanari|Murchison|nakshatra|Nat Cavus|Nepenthes|New Haven|Nia Chaos|Nia Mensa|Nicholson|Nilokeras|Noc Cavus|North Ray|Northport|Oenopides|Palitzsch|pallasite|Pangboche|Pannekoek|Papaleksi|Parkhurst|Patsaev Q|Penticton|Perel'man|Perseus-M|Petermann|Philolaus|Pickering|Pikel'ner|Poczobutt|Poincar√©|pole star|Pomortsev|Priestley|Princeton|Priscilla|Propontis|protostar|PS1-10afx|Quenisset|Raduga 1M|Raspletin|red dwarf|red giant|Rhaeticus|Rima Birt|Robertson|Rosseland|San Marco|satellite|Schickard|Schl√ºter|Schneller|Schr√∂ter|Schroeter|Shijian-6|Shoemaker|Shuleykin|Simpelius|Sniadecki|Sojourner|Sosigenes|South Ray|Steinheil|Steinheim|Sternfeld|Stiborius|Strela-1M|Strela-2M|sub-Earth|supernova|Sylvester|Symphonie|Taruntius|Telstar 3|Terrarium|Thaumasia|Tisserand|TOI 700 d|Topopolis|Trinacria|Trouvelot|Tselina-D|Tseraskiy|Tui Regio|Uvs Lacus|Van Rhijn|Van Vleck|Vitruvius|Volgograd|Von Braun|Wargentin|winonaite|Wollaston|Worcester|Yamal 200|Yantar-2K|Yogi Rock|Zea Dorsa|Zelinskiy|Zenit-4MK|Zenit-4MT|Zsigmondy|≈Ωulanka|2015 VV1|2017 GY8|2019 AB7|2019 CJ3|2019 TG3|2019 VO3|2020 MK4|2020 PA7|2020 PG3|2020 RE7|2020 VF1|2021 GD9|Abenezra|Abulfeda|Aetheria|Al-Bakri|Amazonis|Ameghino|Amenthes|Ammonius|Amontons|Amundsen|Anderson|Andronov|Annegrit|Appleton|Aquacade|Archytas|Artem'ev|Arzachel|Ashbrook|asterism|asteroid|Avicenna|Avogadro|Ayacucho|Baby Ray|Backlund|Badwater|Baillaud|Balandin|Balvicar|Bancroft|Barocius|Barsukov|Baykonyr|Beaumont|Benedict|Bettinus|Bhaskara|Bi Sheng|Birkhoff|Bjerknes|Blackett|Bluebell|Boethius|Bombelli|Bonpland|Bordeaux|Bowditch|Brackett|Bradbury|Brashear|Breislak|Brewster|Bridgman|Brisbane|Butlerov|C. Mayer|Cabannes|Cailleux|Calippus|Camichel|Camiling|Campanus|Campbell|Canberra|Cantoura|Capuanus|Cardanus|Carrillo|Cartosat|Catal√°n|Caventou|Cebrenia|Cecropia|Cerberus|Chadwick|Chalonge|Chandler|Chappell|Charlier|Charlieu|Chekalin|Chimbote|Chinasat|Ching-Te|Ching-tu|Clairaut|Claritas|Clausius|Coblentz|Columbus|Comstock|CondoSat|Congreve|Coprates|Coriolis|Courtney|CUTE-1.7|Cyclopia|Cyrillus|D. Brown|D'Arrest|Da Vinci|Daedalus|Daguerre|Daubr√©e|Davisson|De Vries|Delambre|Delaunay|Delmotte|Delporte|Diad√®me|Dinorwic|Douglass|DS-P1-Yu|Dunhuang|Dunkassa|E√∂tv√∂s|Einstein|Ekspress|Electris|Elektron|Ellerman|Endymion|Epigenes|Eridania|Erlanger|Escorial|Euclides|Euctemon|Eunostos|Evershed|exocomet|Fabbroni|Faustini|Fesenkov|Firmicus|Foucault|Fournier|Franklin|Fredholm|Froelich|G√§rtner|Gadomski|Galdakao|Galilaei|Gandzani|Garavito|Gassendi|Gauricus|Gavrilov|Geissler|Glaisher|Glazenap|Gledhill|Glendore|Golitsyn|Goodacre|Gorizont|Goulburn|Gratteri|Grignard|Grimaldi|Gringauz|Grotrian|Guericke|Gunnison|Guthnick|Hagecius|Hamaguir|Hamilton|Hansteen|Harkhebi|Harpalus|Hartmann|Hatanaka|Heinlein|Heinrich|Heinsius|Hercules|Herschel|Hesiodus|Hesperia|Hevelius|Hiddekel|highland|Hill 305|Himawari|Hippalus|Hirayama|HK Tauri|Horowitz|Horrebow|Horrocks|Huancayo|Huanjing|Humboldt|icy moon|Idel'son|Isidorus|Julienne|Jumpseat|K√§stner|Kamativi|Kamloops|Karpinsk|Karratha|Kathleen|Katoomba|Kepinski|Key Hole|kilonova|Kingston|Kirkwood|Kirsanov|Klaproth|Konoplev|Koreasat|Kramarov|Krasnoye|Kunowsky|L. Clark|Lacchini|Lacrosse|Lagrange|Lahontan|Lampland|landform|Langemak|Langevin|Langmuir|Langtang|Lansberg|Lasswitz|Lawrence|Lebesgue|Legendre|Leibnitz|Leighton|Letronne|Libertad|Lindblad|Lindenau|Lippmann|Liu Hsin|Llanesco|location|Lohrmann|Louville|Lovelace|Lundmark|Lyapunov|M√∂sting|Maestlin|magnetar|Maksutov|Malapert|Malinkin|Manilius|Manzinus|Mariotte|Martynov|Masursky|McDonald|McKellar|Medrissa|Memnonia|Menelaus|Mercator|Meridian|Meteor 3|Meteosat|MetOp-SG|Millikan|Minnaert|Missoula|Mitchell|Miyamoto|Mons Tai|mountain|Murakami|Nadezhda|Nakhlite|Nefed'ev|Neumayer|New Bern|Nicollet|Nikolaev|Noachian|Obruchev|Oceansat|Oppolzer|Orlets-1|Orontius|Oudemans|Palmetto|Palmieri|Panchaia|Parenago|Patricia|Pentland|Peridier|Perrotin|Petavius|Phillips|Pierazzo|Pil√¢tre|Pitiscus|Pizzetti|planetar|planitia|Plaskett|Playfair|Pleiades|Plutarch|Polybius|Polzunov|Poncelet|Pontanus|Poynting|Purkynƒõ|Quetelet|R√©aumur|R√∂ntgen|Raduga-1|Rayadurg|Rayleigh|Regnault|Reimarus|Reinhold|Renaudot|Respighi|Resurs-F|Resurs-P|Reykholt|Reynolds|Rhysling|Riccioli|Richards|Robinson|Rocknest|Rothmann|Sabatier|San Juan|Santa Fe|Santbech|Sarabhai|SATCOMBw|Saussure|Savannah|Scaliger|Sch√∂ner|Scheiner|Schiller|Schubert|Schuster|scopulus|Scoresby|Sechenov|Seeliger|Seleucus|Semeykin|Seminole|Shahinaz|Sharonov|Shatalov|Shatskiy|Sherlock|Shi Shen|Sikorsky|Sirsalis|Sisakyan|Smithson|Snellius|Somerset|St. John|St√∂fler|St√∂rmer|Stebbins|Stevinus|Stoletov|Stratton|Strela-1|Strela-2|Strela-3|Subbotin|subgiant|Surveyor|Svedberg|Sverdrup|T. Mayer|Tacchini|Tannerus|Tarrafal|Tecolote|Thiessen|Tian Shi|Tiantong|Tibrikot|Timbuktu|Tiselius|Tolansky|Tombaugh|Trumpler|Tuskegee|Tyuratam|U Pegasi|Uchronia|ureilite|Valverde|Van Gent|Van Serg|Van Wijk|Vanguard|vastitas|VDK IIIa|VDK IIIb|VDK VIII|VDK XIII|Verlaine|Vertregt|Vesalius|Victoria|Virtanen|Vishniac|Volterra|Wapowski|Waterman|Weinbaum|Wichmann|Wiechert|Williams|Windfall|Winthrop|Xenophon|Yakovkin|Yamamoto|YGKOW 98|Yorktown|Z√∂llner|Zasyadko|Zenit 2M|Zenit 6U|Zenit-4M|Zephyria|Zucchius|1994 TA|2019 WJ|ADEOS I|Aepinus|Agassiz|Agrippa|Alekhin|Alhazen|Almanon|Am star|Angrite|Anuchin|Anville|Apianus|Arandas|Arcadia|ArduSat|Artemis|Asclepi|ataxite|Aubrite|Aureole|Ausonia|Babakin|Babbage|Babcock|Bacolor|Baltisk|Banting|Barbier|Barnard|Bartels|Be star|Beketov|Belyaev|Bentham|Bentong|Bergman|Berkner|Berlage|Bernard|Berosus|Berseba|Bilharz|Bingham|Blazhko|blitzar|Bombala|Bouguer|Brayley|Brenner|Briault|Bristol|Bront√´|Brouwer|Brunner|Buisson|Burnham|Byrgius|Cai Lun|Calamar|Camargo|Camelot|Cameron|Canillo|Cankuzo|Capella|Cardona|Carlini|Cartago|Casatus|Cassini|Castril|Cefal√π|Celsius|centaur|Cepheus|Cerulli|Chaffee|Challis|Chapais|Chapman|Charles|Chaucer|Chladni|Choctaw|Cidonia|Clavius|Cluster|Cochise|Coimbra|Collins|Colombo|COMPASS|Compton|Conches|Concord|Corinto|Corozal|Coulomb|Cr√ºger|Cremona|Crivitz|Crookes|Crozier|CubeSat|Curtius|Cusanus|Cydonia|Cypress|Cysatus|Dampier|Daniell|De Vico|Delisle|Demonax|Denning|Deseado|Deutsch|Dia-Cau|Diacria|Diderot|Doerfel|Dollfus|Dollond|Doppler|Drebbel|Dromore|DS-K-40|Du Toit|Dubyago|Ehrlich|Eijkman|Eimmart|Ekran-M|Ellison|Ellsley|Elysium|Eucrite|Eudoxus|Fairouz|Faraday|FASTRAC|Fechner|Fedorov|Fengyun|Fersman|Fibiger|Firsoff|Fischer|Fitzroy|Fleming|Flock-1|fluctus|Fontana|Forseti|Fourier|Freedom|Fridman|Fryxell|Funchal|G. Bond|Gagarin|Galvani|Gambart|Ganskiy|Gardner|Geminus|GG Lupi|Gilbert|glacier|Glauber|Glenelg|GLONASS|Glushko|GM Lupi|Goddard|Golovin|Gr√≥jec|Grachev|Greaves|Greeley|Gregory|Grissom|Guaymas|Gyld√©n|H√©lios|Hainzel|HaiYang|Haldane|Halfway|Hamelin|HaradÃ¶|Harding|Harriot|Hartwig|Haworth|Hayford|Heimdal|Helberg|Helicon|helipad|Helmert|Henbury|Hendrix|Hermann|Hermite|Heymans|Hilbert|Hohmann|Hopmann|Horatio|Hornsby|Houssay|Houston|Houzeau|Huggins|Humason|Husband|Huygens|Hyginus|Hypatia|II Lupi|IM Lupi|Ingalls|Ireland|Irharen|Jackson|Janssen|Jenkins|Jodrell|Jojutla|Kalocsa|Kantang|Kapteyn|Kartabo|Kashira|Kasimov|Kearons|Kekul√©|Keldysh|Kerbtal|KH 8-20|Khanpur|Khujirt|Kidinnu|Kinkora|Kircher|Kisambo|Kolonga|Komarov|KOMPSAT|Korolev|Kosberg|Koshoba|Kozyrev|Kramers|Krasnov|Kreiken|Krieger|L dwarf|Lachute|Lacroix|Lagalla|Lagarto|Lalande|Lam√®ch|Lamarck|Lambert|Langley|Lassell|Laveran|Leavitt|Lebedev|Lehmann|Leleque|Lemuria|Lepaute|Liberta|Licetus|Lindsay|lingula|Lipskiy|Lismore|Littrow|Lockyer|Lodygin|Lorentz|Lowbury|Lubbock|Lunar X|M√§dler|M√∂bius|M√ºller|Maclear|Maggini|Maginus|MAMBO-9|Mandora|Manners|Maraldi|Marbach|Marconi|Mariner|Marinus|Marisat|Maunder|Maxwell|Mazamba|McClure|McMurdo|McNally|Meggers|Meitner|Melissa|Mellish|Mendota|Merrill|Messala|Messier|Michael|Milford|Millman|Mitchel|Moiseev|Moissan|Molniya|moonlet|Morella|Moretus|Morozov|Moseley|Mouchez|Moulton|N√∂ther|Nagaoka|Naonobu|Nasmyth|Natasha|Naumann|Neander|Neujmin|Newcomb|Newport|Nicolai|Nielsen|Niesten|Nijland|Nikolya|Nipigon|Niquero|Nishina|NOAA-15|NOAA-16|NOAA-17|NOAA-18|NOAA-19|Noachis|Numerov|OB star|Ochakov|Oersted|Okhotsk|Okotoks|Olivier|Olympia|Onizuka|Orbcomm|orbiter|Oreol 3|Ortygia|Ostwald|Ottumwa|outcrop|Palikir|Parsons|Paschen|Pasteur|Patsaev|Perrine|Persona|Petzval|Philips|Phlegra|Pingr√©|Pirquet|Pitatus|planemo|Plant√©|plateau|Plinius|Plummer|plutino|plutoid|Poinsot|Poisson|Pollack|Polotsk|Pompeii|Prandtl|Proclus|Proctor|Prognoz|proplyd|Puiseux|Purbach|Pytheas|Quivira|Quthing|Raimond|Ramsden|Rankine|Razumov|Repsold|Riccius|Riemann|Ritchey|Roberts|Rongxar|Rowland|Rumford|Runanga|Russell|Rydberg|Saenger|Samos-F|Sampson|Sandila|Sanford|Santaca|Saravan|Saunder|Scandia|Scheele|Schmidt|Schöner|Schwabe|Sednoid|Selevac|serpens|Seyfert|Shapley|Shawnee|Slipher|SMART-1|Soochow|Sp√∂rer|Srƒ´pur|Stadius|statite|Stearns|Steklov|Stetson|Stewart|Sundman|Swanage|Szilard|Tacitus|Tacquet|Tai Wei|Tarakan|Tebbutt|Telstar|Tharsis|Theiler|Thermia|Thomson|Tianjin|Tignish|Timaeus|Tintina|TIROS-N|Toconao|Tooting|Townley|Tralles|Trident|Triolet|Trumpet|Tsander|Tsegihi|Tselina|Tsikada|Tsiklon|Tsinger|Tsukuba|Tugaske|twotino|Tyndall|Vaughan|Vavilov|VDK III|VDK VII|VDK XII|VDK XIV|VDK XVI|Ventris|Vestine|Victory|Virchow|Vitello|Viviani|Voeykov|volcano|W. Bond|W√∂hler|Wallace|Wallach|Wallops|Wallula|Walther|Wan-Hoo|Wassamu|Wegener|Whewell|Whipple|Wicklow|Wilhelm|Wilkins|Wilsing|Winkler|Winlock|Winslow|Woltjer|Woolgar|Woomera|Yangel'|Zanstra|Zenit 8|Zenit-2|Zenit-4|Zernike|Zutphen|√ëuSat|≈åmura|A type|Abetti|Acosta|Aeolis|Afekan|Airy-0|Aitken|Alamos|Albany|Albert|Aldrin|Alitus|Andapa|Anders|Andƒõl|Apollo|Arabia|Aratus|Argyre|Arnold|Asimov|Atwood|Auwers|Auzout|Avarua|Aveiro|Azophi|Aztlan|B type|Bailly|Balboa|Baldet|Ballet|Balmer|Baltia|Barkla|Barrow|Bars-M|Bato≈ü|Batoka|Baucau|Beacon|Beagle|Behaim|Bellot|Beloha|Beltra|Belyov|Beruri|Bessel|Bhabha|Bigbee|blanet|blazar|Blitta|Blunck|Bobone|Bolyai|Bopolu|Borman|Boulia|Bozkir|Braude|Bridge|Briggs|Broach|Buffon|Bulhar|Bunnik|Bunsen|Burton|C√°diz|Ca√±as|Cabeus|Cairns|Cajori|Camiri|Campos|Canala|Candor|Cangwu|Cannon|Cantor|canyon|Carlos|Carnot|Carrel|Cartan|Carver|Casius|Catota|Cauchy|Caxias|Cayley|Cefalù|Cexing|Chalce|Chaman|Chappe|chasma|Chawla|Chinju|Chryse|Cichus|Cilaos|Circle|Clerke|Cobalt|Cobres|Col√≥n|colles|Comrie|Condon|Cooper|Corona|Couder|crater|Crocco|Culter|Curtis|Cuvier|Cyrano|Dalton|Danjon|Darney|Darvel|Darwin|Davies|Dawson|De Roy|Dechen|Dejnev|Dessau|Dilmun|Dogana|Domoni|Donati|Donner|Double|Draper|Dreyer|Dryden|Dukhan|Dulovo|Dun√©r|E type|Easley|Eckert|Edison|Elorza|Erebus|Espino|F type|F√©nyi|Falcon|Fastov|Fenagh|Fensal|Fermat|Finsch|Finsen|Firsov|Fizeau|Florey|Foster|Fowler|Franck|FV Sco|galaxy|Galois|Gamboa|Gander|Gaston|Gastre|Geiger|Gerard|Ginzel|Glazov|Golden|Gonets|graben|Grójec|Groves|Gunjur|Hadley|Halley|Hansen|Harden|Harlan|Harold|Harris|Harvey|Hashir|Haskin|Hausen|Hellas|Henyey|Hevesy|Holden|Holmes|Hommel|Hottah|HR Peg|Hubble|Hunten|Hussey|Hutton|Huxley|Icaria|Icarus|Ideler|Inuvik|Isabel|Izendy|Jacobi|Jampur|Jamuna|Jansen|Jansky|Jarvis|Jenner|Jez≈æa|Jezero|Jijiga|Joliot|K√∂nig|Kachug|Kaiser|Kakori|Kalpin|Kamnik|Kampot|Karima|Karrer|Karshi|Karzok|Kasabi|Kasper|KazSat|Keeler|Kepler|Khurli|Kibuye|Kifrƒ´|Kimura|Kipini|Knobel|Kocher|Kontum|Kosmos|Kostya|Kourou|Krafft|Krupac|Krylov|Kugler|Kuiper|Kumara|Kushva|La Paz|Labria|LAGEOS|Lamont|Landau|lander|Larmor|Layl√°|Leakey|Lenard|Leonid|Leonov|Lexell|Li Fan|Liebig|Lilius|Linn√©|Lipany|Lisboa|Locana|Lodwar|Lomela|Lorica|Louise|Lovell|Lowell|Lucaya|Lucian|Ludwig|Luna 9|Luther|Mackin|macula|Mädler|Madrid|Magadi|Magnum|Magong|Mairan|Majuro|Mallet|Manuel|Manzƒ´|Marius|Markov|Mars 6|Martin|Matara|Maya-1|McAdie|McCool|McMath|McNair|Mellit|Mendel|Menrva|Mentor|Menzel|Meteor|Metius|Micoud|Miller|Mineur|Mirtos|Moanda|Mohawk|Moigno|Mojave|Moltke|Monira|Moreux|Morley|Müller|Murgoo|Murray|Mystis|Nakusp|Nansen|Naruko|Nassau|Naukan|Nearch|nebula|Nectar|Negele|Negril|Neison|Nereus|Nernst|NetSat|Neukum|Newton|Niepce|Njesko|NOAA-6|NOAA-B|Nobile|Nobili|Nonius|Norman|Novara|NZ Peg|Oberth|Ocampo|Oglala|Olbers|Olcott|Olenek|Olydri|OO Peg|Oraibi|Oresme|Orinda|Orlets|Osiris|Ostrov|P√∫nsk|PAGEOS|Palana|Palapa|Palisa|Pallas|Paneth|Parrot|Pascal|patera|Pavlov|Pawsey|Peirce|Perkin|Persbo|Peters|Petrie|Petrov|Pettit|Phedra|Phison|Piazzi|Picard|Pictet|Pinglo|Planck|planet|planum|Platte|Pogson|Porter|Porvoo|Powell|PPl 15|Prager|Proton|Pulawy|pulsar|Pursat|Q star|quasar|Quines|R√∂mer|Raduga|Ramsay|Reiner|Resnik|Resurs|Reutov|Rheita|Riedel|Rincon|Ritter|Robert|Rohini|Roseau|Rossby|RS Sgr|Rudaux|S√∂gel|Sabine|Saheki|Salaga|Sangar|SAOCOM|Sarton|Satcom|Savich|Schorr|Scobee|Seares|Secchi|Sefadu|Segers|Segner|Seidel|SELENE|Seneca|Senkyo|Shaler|Shambe|Shardi|Shioli|Sibiti|Sinlap|Sinton|Sirius|Sitrah|Skynet|Slater|Slocum|Soffen|Solano|Solrad|Soraya|Souris|Spudis|Srīpur|Stefan|Stella|Stokes|Stoney|Strabo|Street|Strela|Struve|Stubby|Sumgin|Sumner|SURCAL|Swasey|Syncom|T√°bor|Taejin|Talbot|Taltal|Tanais|Tarata|Tarsus|Taylor|Taytay|Tempel|Thales|Thebit|tholus|Tikhov|Tiling|Timaru|Titius|Tivoli|Tomari|Tomini|Torbay|Tors√∂|Troika|trojan|Tucker|Tungla|Turner|UGC 22|Umatac|unnova|Utopia|V√§t√∂|Valera|Valier|valley|vallis|VDK II|VDK IV|VDK IX|VDK VI|VDK XI|VDK XV|Vernal|Vil'ev|Virrat|Vivero|Vol'sk|Volkov|Vortex|Vostok|Wabash|Walker|Walter|Warner|Waspam|Watson|Weigel|Weinek|Werner|Westar|Wexler|Wiener|Wilson|Woking|Wright|Wukari|Xainza|Xanadu|Xanthe|Yakima|Yalata|Yalgoo|Yantar|Yaogan|Yegros|Yerkes|Yungay|Zarand|Zaranj|Zeeman|Zhigou|Zhinyu|Zi Wei|Zilair|Zinner|Zvezda|Zwicky|Abbot|Achar|Adams|Adiri|Aeria|AEROS|Aktaj|Alden|Alder|Alnif|Aloha|Alter|Alvin|Amici|Aniak|Arago|arcus|Argas|Arica|Arima|Arnon|Artik|Asada|Aspen|Aston|Atlas|Avery|Avire|Azusa|B√©la|B√ºrg|Baade|Bacht|Baily|Balta|Bamba|Banff|Barth|Basin|Batoş|Bayer|Bazas|Beals|Belet|Bench|Berry|Betio|Biela|Billy|Bison|Black|Blagg|Bland|Bliss|Blois|Bluff|Bogia|Bogra|Boola|Boole|Borda|Borel|Boris|Borya|Bosch|Bowen|Boyle|Bragg|Bronk|Brown|Bruce|Brush|Bunge|Butte|Byala|Byske|Cádiz|Cajal|Calbe|Cañas|Canso|Capen|Carol|cavus|Cayon|Chafe|Chant|Chauk|Chefu|Chive|Choyr|Cinco|Clark|Clogh|Clova|Cluny|comet|Conon|Cooma|Corby|Creel|Crewe|Crile|Cross|Cruls|Curie|Dante|Dawes|Debes|Debus|Debye|Delia|Delta|Deluc|Dersu|Dewar|Diana|Dilly|Dingo|Dison|Dixie|Dokka|Donna|Downe|Drude|Dubki|Dufay|Dugan|Dyson|Dzeng|Eagle|Eddie|Edith|Egede|Ehden|Eilat|Ekran|Elath|Elbow|Elger|ELISA|Elmer|Elvey|Emden|Emory|Encke|EPE-A|EPE-B|EPE-C|EPE-D|Esira|Espin|Euler|Evans|Fabry|Faith|Falun|Fancy|Farim|Fauth|Felix|Fermi|Flora|Focas|Foros|fossa|Franz|Freud|Frost|fusor|Gagra|Galap|Galen|Galle|Gamow|Gandu|Gardo|Garni|Gator|Gauss|Geber|Gehon|Gibbs|Gioja|GIOVE|Glide|Globe|Godin|Gokwe|Golgi|Gould|Grace|Graff|Grave|Green|Grigg|Grove|Guest|Gulch|Gusev|Guyot|Gwash|Haber|Hagen|Halba|Hanno|Haret|Hawke|Healy|Hedin|Henry|Heron|Hertz|HiROS|Honda|Hooke|Il'in|Index|Injun|Innes|Ioffe|Irbit|Isaev|Istok|Izsak|J√∂rn|JCSAT|Jeans|Jehan|Jerik|Jones|Jos√©|Joule|Jumla|Kagul|Kalba|Kanab|Kandi|Kansk|Kasra|Kayne|Keren|Keul'|KƒÅid|KH-12|Kholm|Kiess|Kimry|Kinau|Kinda|Kirch|Klein|Klute|Kolya|Kopff|Korph|Kotka|Kribi|Krogh|Kufra|Kular|Kulik|Kumak|Kundt|Kunes|labes|lacus|Lam√©|Lamas|Lapri|Lemgo|Lents|Lenya|Lewis|Liais|Libya|Linda|Linpu|Lipik|Litke|Livny|Loewy|Lohse|Lonar|Longa|Louth|Luzin|Lydda|Lyell|Lyman|Mafra|Malyy|Manah|Manti|Marca|Marci|Marth|Martz|Mason|Maury|Mavis|Meget|mensa|Meroe|Meton|MetOp|Mills|Milna|Milne|MiTEx|Mitra|Mliba|Momoy|Monge|Moore|Moroz|Morse|Muara|Mutch|Mutus|Nardo|Naryn|Navan|Nazca|nCube|Necho|Neive|Neper|Never|Neves|Nhill|Nimiq|Nitro|Nobel|Noord|Nqutu|Nu≈°l|Nutak|O'Day|Obock|Ohara|Okean|Omega|Opelt|Ophir|Orlov|ORS-5|Osama|OSCAR|Osman|Oyama|Palos|palus|Paros|Parry|Pauli|Paxsi|Peary|Pease|Pebas|Peixe|Petit|Pi√±a|Pital|plain|Plana|Plato|Podor|Polar|Poona|Popov|Poppy|Porth|Prinz|PROBA|Pupin|Pylos|QibƒÅ|Quick|Quill|Quorn|Racah|Radau|Rakke|Raman|Ramon|Rauch|Rauna|Rayet|Recht|Rengo|Resen|Reuyl|Revda|Ricco|Rigel|rille|Rimac|RISAT|Rocca|Rocco|Roche|Roddy|Romeo|Romny|Rosse|Ruhea|Runge|rupes|Ryder|Rynin|Rynok|Rypin|Sagan|Samir|Sarno|Satka|Scott|Sebec|Sevel|Sfera|Sharp|Shayn|Short|Sibut|Sigli|Sinai|Sinas|Sinda|Singa|Sinop|sinus|Sitka|Slava|Smith|Soddy|Sokol|South|Spook|Spurr|Stark|Stege|Stein|Steno|Stobs|Suata|Sucre|Suess|Sulak|Susan|Swann|Swarm|Swift|Syria|Tabou|Taizo|Talas|Talsi|Tarma|Tavua|Taxco|taxon|Tempe|Tepko|Terby|terra|Tesla|Tharp|Thiel|Thila|Thira|Thoth|Thule|TIROS|Titov|Tivat|Tokko|Tokma|Tolon|Tombe|Torup|Tuapi|Tumul|Turbi|Turma|Tycho|Ukert|Umbra|Vaals|Vaduz|Valga|Vasya|VDK I|VDK V|VDK X|Veles|Verne|Viana|Vieta|Vitya|Vlacq|Vogel|Volta|Wafra|Wahoo|Wajir|Wargo|Warra|Watts|Weber|Weert|Weiss|Wells|White|Wildt|Wiltz|Wirtz|Wulai|Yamal|Yaren|Yebra|Yoshi|Young|Zagut|Zenit|Zongo|Zumba|Zunil|Zupus|Aaru|Aban|Abbe|Abel|Airy|Ajon|Akis|Alan|Albi|Alga|Amos|Ango|Angu|Anik|Apia|Argo|Arta|Asau|Auce|Auki|Avan|Azul|Back|Baco|Bada|Bahn|Ball|Banh|Baro|Bawa|Beag|Beer|Bell|Belz|Bend|Bhor|Biot|Bira|Birt|Bise|Bled|Bode|Bohr|Bole|Bond|Born|Boru|Bose|Boss|Bree|Buch|Buta|Byrd|Cave|Cheb|Chia|Chom|Chur|Cone|Cook|Cori|Cost|crag|Cray|Daan|Daet|Dale|Daly|Dana|Dank|Davy|Deba|Dein|Dese|Doba|Doon|Dove|Dowa|Dune|Dush|Eads|Echt|Edam|Edom|Eger|Elim|EROS|Erro|Ewen|Faqu|Faye|Flag|Flat|Floq|Fram|Gaan|Gale|Gali|Gals|Galu|Gari|Garm|Gasa|Gena|Gill|Goba|Goff|Gold|Gore|Gori|Greg|GSAT|Guir|GZ B|Hahn|Hale|Hall|Halo|Hano|Hase|Hayn|Head|Hegu|Heis|Hell|Hess|Hƒ´t|hill|Hind|Hogg|Hope|Hume|Iazu|Igal|Igor|Ikej|Imgr|Inta|Ipmo|Isil|Isis|Ivan|Jama|Jiji|Joly|Jomo|Jörn|Kane|Kant|Kaup|Kem'|Kies|King|Kira|Kirs|Kita|Kiva|Koch|Koga|Kong|Kuba|Kuhn|Lade|Lamb|Land|Lane|Lara|Last|Laue|Lebu|Leuk|Lick|Link|Lins|Loja|Loon|Lota|Loto|Love|LQ01|LQ02|LQ03|LQ04|LQ05|LQ06|LQ07|LQ08|LQ09|LQ10|LQ11|LQ12|LQ13|LQ14|LQ15|LQ16|LQ17|LQ18|LQ19|LQ20|LQ21|LQ22|LQ23|LQ24|LQ25|LQ26|LQ27|LQ28|LQ29|LQ30|Luba|Luch|Luck|Luga|Luki|Luqa|Lyot|Mach|Mago|Main|Mari|Mary|Mees|Mega|Mena|mesa|Mila|Moab|Moni|mons|Moon|Moss|Naar|Naic|Nain|Naju|Nath|Nema|Nepa|Nier|NOAA|Noma|nova|Nune|Nunn|Ofeq|Oken|Olom|Onon|Oxus|P-11|P√°l|Pabo|Paks|Peek|Peta|Phon|Pica|Piyi|Plum|Pons|Poti|Prao|Puyo|Rabe|Raga|Rahe|Rana|Raub|Ravi|Redi|Ribe|rift|Ritz|Rong|Rosa|Ross|Rost|Ruby|Ruth|Ruza|S≈´f|Sabo|Saha|Sarh|Sarn|Sauk|Selk|Sevi|Sfax|Sian|Sibu|Sita|SPOT|Spry|Spur|star|Ston|STRV|Styx|Surt|Tala|Talu|Tame|Tamm|Tara|Taza|Tejn|Telz|Tem'|Thom|Thor|Tile|Tiwi|Toro|Troy|Trud|Tsau|Tura|Ubud|Ulya|unda|Urey|US-A|US-P|Utan|Uzer|Vaux|Vega|Vera|Very|Vils|void|Voza|Watt|Webb|West|Weyl|Wien|Wink|Wolf|Wood|Wyld|Yala|Yoro|Yuty|Zach|Zeno|Zuni|Ada|Aki|Ann|Apt|Ayr|Bak|Bam|Bar|Bok|Bor|Can|Cue|Dag|Das|Eil|EKS|Ely|Esk|Fox|Gah|Gan|Gol|Gum|Ham|Ian|IIC|Ina|Ins|Jal|Jen|Joy|Kaj|Kao|Kaw|Kin|Kok|Koy|Ksa|Laf|Lar|Lau|Lee|Lev|Ley|Lod|Los|Mee|Mie|Mut|Nan|Nif|O3b|Ohm|Ome|Ore|Pau|Sat|Say|Soi|Tak|Ulu|Urk|Vik|Voo|Wau|Wer|Xui|Yar|Yat|Zir
    '''
    re_wiki_vocab = regex.compile(r'\b(%s)\b' % wiki_vocab)

    def extract_top_keywords(self, text):
        """

        :param text:
        :return:
        """
        matches = []
        for match in self.re_wiki_vocab.findall(text):
            if isinstance(match, tuple):
                match = [item for item in match if item]
            else:
                match = [match]
            matches += match
        return list(set(matches))


class NASAWrapper():

    def forward(self, excerpt):
        """

        :param excerpt:
        :return:
        """
        url = config['PLANETARYNAMES_PIPELINE_NASA_CONCEPT_URL']
        payload = {
            "text": excerpt,
            "probability_threshold": 0.5,
            "topic_threshold": 1,
            "request_id": "example_request_id"
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            result = response.json()['payload']
            sti_keywords = result.get('sti_keywords',[[]])[0]

            # extract the 'unstemmed' from 'sti_keywords'
            sti_keywords = [kw['unstemmed'].lower() for kw in sti_keywords]
            return sti_keywords
        else:
            logger.error(f"From Nasa Concept status code {response.status_code}")
        return []


class ExtractKeywords():

    def __init__(self, args: EntityArgs):
        """

        """
        self.args = args
        self.spacy = SpacyWrapper()
        self.yake = YakeWrapper()
        self.wiki = WikiWrapper()
        self.tfidf = TfidfWrapper()
        self.nasa = NASAWrapper()

    def forward(self, excerpt, num_keywords=16):
        """
        three ways to identify top keywords from the excerpt and then merge
        1- entities identified by Spacy
        2- top keywords identified by yack
        3- match with astronomy object vocabs from wiki
        if any of the keywords is a phrase containing the feature_name, quit

        :param excerpt:
        :param num_keywords:
        :return:
        """
        feature_types = set([type.lower() for type in [self.args.feature_type, self.args.feature_type_plural]])
        feature_name = [self.args.feature_name.lower()]

        spacy_keywords = list(set([token.lower() for token in self.spacy.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(spacy_keywords, feature_name, feature_types):
            logger.info('SpaCy identified a phrase that included feature name. Excerpt filtered out.')
            return []
        yake_keywords = list(set([token.lower() for token in self.yake.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(yake_keywords, feature_name, feature_types):
            logger.info('Yake identified a phrase that included feature name. Excerpt filtered out.')
            return []
        wikidata_keywords = list(set([token.lower() for token in self.wiki.extract_top_keywords(excerpt) if token != self.args.feature_name]))
        if not self.verify(wikidata_keywords, feature_name, feature_types):
            logger.info('Wikidata keyword has a phrase that includes feature name. Excerpt filtered out.')
            return []

        # find tokens shared between spacy and yake
        spacy_yake_shared = []
        if spacy_keywords and yake_keywords:
            for keyword in yake_keywords.copy():
                for entity in spacy_keywords:
                    if keyword in entity:
                        spacy_yake_shared.append(keyword)
                        spacy_keywords.remove(entity)
                        yake_keywords.remove(keyword)
                        break

        # add remaining tokens from wikidata if available
        num_keywords_from_wiki = num_keywords - len(spacy_yake_shared)
        keywords = list(set(spacy_yake_shared + wikidata_keywords[:num_keywords_from_wiki]))

        # add more keywords to be returned if not have enough keywords yet
        if len(keywords) < num_keywords:
            keywords += wikidata_keywords[num_keywords_from_wiki:]
            keywords = list(set(keywords))

        # if still dont have enough keywords
        # first combine the two lists of spacy and yake, alternate between them, and then add them to the returned token list
        if len(keywords) < num_keywords:
            if spacy_keywords and yake_keywords:
                count = min(len(spacy_keywords), len(yake_keywords))
                combined = [''] * (count * 2)
                combined[::2] = spacy_keywords[:count]
                combined[1::2] = yake_keywords[:count]
                combined.extend(spacy_keywords[count:])
                combined.extend(yake_keywords[count:])
                keywords += combined
            elif spacy_keywords:
                keywords += spacy_keywords
            elif yake_keywords:
                keywords += yake_keywords
        keywords = list(set(keywords))[:num_keywords]

        # need to at least return 2/3 of keywords asked for, otherwise return nothing
        if len(keywords) >= num_keywords * 0.66:
            return keywords

        logger.info('Not enough keywords were extracted. Excerpt filtered out.')
        return []

    def forward_doc(self, doc, vocabulary, usgs_term, num_keywords=20):
        """

        :param doc:
        :param vocabulary:
        :param usgs_term:
        :return:
        """
        re_vocabulary = regex.compile(r'(?i)\b(?:%s)\b' % '|'.join(vocabulary))
        tfidf_keywords = self.tfidf.extract_top_keywords(doc)
        count_matches = len(list(set([token for token in tfidf_keywords if re_vocabulary.search(token)])))
        # when usgs_term, planetary related, need at least one matched token to among the vocabulary
        # to get included for processing
        if usgs_term and count_matches >= 1:
            return tfidf_keywords[:num_keywords]
        # for non usgs_term, non planetary related, non of the matched tokens should be among the vocabulary
        # to get included for processing
        if not usgs_term and count_matches == 0:
            return tfidf_keywords[:num_keywords]
        return []

    def forward_special(self, excerpt):
        """

        :param excerpt:
        :return:
        """
        return self.nasa.forward(excerpt)

    def verify(self, keywords, feature_name, feature_types):
        """
        verify identified keywords/phrases
        if both feature name and feature types are part of a keyword phrase or
        neither have appeared in the identified keyword phrases all good -> proceed,
        return True
        otherwise if feature name is part of a phrase with another token, that means
        the feature name has a context other than planetary that we are trying to id,
        in that case we have to case stop, return False

        :param keywords:
        :param feature_name:
        :param feature_types:
        :return:
        """
        for keyword in keywords:
            keyword_tokens = set([k.lower() for k in keyword.split()])
            # we need to compare multi token keywords
            if len(keyword_tokens) == 1:
                continue
            # is keyword either the feature name or feature types
            matched_feature_name = len(keyword_tokens.intersection(feature_name))
            matched_feature_types = len(keyword_tokens.intersection(feature_types))
            if matched_feature_name != matched_feature_types and matched_feature_name >= 1:
                return False
        return True