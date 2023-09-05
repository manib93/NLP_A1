# models.py
import math
import random

from sentiment_data import *
from utils import *

from collections import Counter
import numpy as np

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def __init__(self):
        self.indexer = Indexer()

    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer()
        #self.words_list = []

    def size(self):
        return 15000

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        if type(sentence) == list:
            temp_sentence = sentence
        else:
            temp_sentence = sentence.words
            #for word in sentence.words:
            #    self.words_list.append(word)

        items_to_remove = []
        final_words = []
        for word in temp_sentence:
            word = word.lower()
            if word not in items_to_remove:
                final_words.append(word)
                items_to_remove.append(word)
        index_of_words = [self.indexer.add_and_get_index(word, add=add_to_indexer) for word in final_words]

        #Initiating the feature size to be 10,000 words
        feature = np.zeros(self.size())
        for i in index_of_words:
            if i <self.size():
                feature[i] +=1
        return feature


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer()


    def size(self):
        return 75000

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        if type(sentence) == list:
            temp_sentence = sentence
        else:
            temp_sentence = sentence.words

        items_to_remove = []
        final_words = []
        for word in temp_sentence:
            word = word.lower()
            if word not in items_to_remove:
                final_words.append(word)
        index_of_words = []
        for i in range(len(final_words)-1):
            z = final_words[i] + '|' + final_words[i+1]
            index_of_words_return = self.indexer.add_and_get_index(z, add=add_to_indexer)
            index_of_words.append(index_of_words_return)
        #index_of_words = [self.indexer.add_and_get_index(final_words[i]+'|'+ final_words[i+1]), add=add_to_indexer) for i in range(len(final_words)-1)]

        #Initiating the feature size to be 10,000 words
        feature = np.zeros(self.size())
        for i in index_of_words:
            if i <self.size():
                feature[i] +=1
        return feature

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = Indexer()

    def size(self):
        return 14000

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:

        if type(sentence) == list:
            temp_sentence = sentence
        else:
            temp_sentence = sentence.words

        punctuations = '''()-[]{};:'"\,<>./?@#$%^&*_~'''
        punctuations = list(punctuations)

        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                     "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
                     "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
                     "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
                     "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                     "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
                     "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
                     "out", "on", "off", "over", "under", "again", "then", "once", "here", "there", "when",
                     "where", "so", "than", "can", "will", "just", "should"]

        low_frequency_words = ['diamond', 'unwary', 'extremists', 'lawn', 'partnership',
                        'herring', 'reel\\/real', 'dichotomy', 'pursued', 'less-than-thrilling',
                        'affectation-free', 'stoner', 'deconstruction', 'bedevilling', 'upends', 'dilettante',
                        'Arguably', 'entertainments', 'brats', 'contemptible', 'imitator', 'SNL', '8-year-old',
                        'money-oriented', 'guilt-free', 'Undoubtedly', 'tattoos', 'torpedo', 'scripters', 'Oscars',
                        'Kiarostami', 'labour', 'chiaroscuro', 'collage', 'dim', 'echo', 'O.', 'Gift', 'Magi',
                        'relocated', 'scuzzy', 'NYC', 'inflate', 'mileage', 'stray', 'phonograph', 'recognized',
                        'bilked', 'insistence', 'opting', 'Seigner', 'naturalism', 'Revenge', 'Nerds', 'restage',
                        'bathtub', 'cheapen', 'food-for-thought', 'instructive', 'placed', 'swashbucklers', 'Corbett',
                        'Zwick', 'Quotations', 'scratch', 'joyful', 'solo', 'disturbance', 'sole', 'A.C.', 'McAdams',
                        'Since', 'hookers', 'sub-formulaic', 'pithy', 'boxes', 'stones', 'shameful', 'tease',
                        'video-viewing', 'uninhibited', 'Daring', 'raffish', 'dramatization', 'ZigZag', 'densely',
                        'renowned', 'Illiterate', 'inert', 'pet', 'Marker', 'essayist', 'ate', 'Reeses', 'peanut',
                        'butter', 'Astonishingly', 'informed', 'wireless', 'laboratory', 'Westbrook', 'reinvigorated',
                        'Berkley', 'flopping', 'dolphin-gasm', 'sleekness', 'backed', 'facade', 'movie-biz',
                        'uncreative', 'anniversary', 'retrospectively', 'Hunk', 'destructive', '\\*\\*\\*\\*',
                        'disassociation', 'Raimondi', 'risk', 'near-fatal', 'Spreads', 'commune', 'F', 'rom',
                        'blazingly', 'hurried', '1959', 'Godzilla', 'commenting', 'pyschological', 'knocks', 'Build',
                        'robots', 'Mystery', 'Science', 'hotter', 'Georgia', 'asphalt', 'wonderment', 'vagueness',
                        'exasperating', 'rollerball', 'sanitised', 'Chilling', 'twenty-first', 'Petri',
                        'well-characterized', 'telephone', 'impassive', 'Brims', 'accident-prone', 'Mastering',
                        'formidable', 'arithmetic', 'cameras', 'flood', 'agitator', 'Strong', 'requires',
                        'oh-so-important', 'boardwalk', 'self-amused', 'iteration', 'creepiest', 'developmentally',
                        'disabled', 'redeem', 'spectrum', 'well-deserved', 'stylists', 'zoning', 'ordinances',
                        'dabbles', 'page-turning', 'frenzy', 'Holmes', 'major-league', 'underachiever', 'edifying',
                        'revolutionary', 'regimen', 'sleeping', 'stress-reducing', 'scoring', 'assets', 'commend',
                        'Mika', 'Anna', 'Mouglalis', 'spot-on', 'burr', 'peels', 'Third', 'yeah', 'courtesy', 'Pogue',
                        'Yale', 'grad', 'previously', 'Skulls', 'excepting', 'encountered', 'Superb', 'somnambulant',
                        'assigned', 'Afterschool', 'Versace', 'thirty-three', 'feathers', 'sandbox', 'affluent',
                        'damsel', 'distress', 'bully', 'overdone', 'establishing', 'lapping', 'wrecks', 'missive',
                        'thrilled', 'compulsive', 'Tolstoy', 'groupies', 'unstable', 'aid', 'Indeed',
                        'shock-you-into-laughter', 'Dadaist', 'Band', 'Pick', 'Pieces', 'famed', 'realities',
                        'all-inclusive', 'uptight', 'bores', 'Compellingly', 'self-flagellation', 'minutiae',
                        'Brainless', 'fifties', 'teen-gang', 'rough-trade', 'homo-eroticism', 'Aiello', 'mumbles',
                        'espoused', 'envelops', 'fluttering', 'stammering', 'ruins', 'wreaked', 'compromised',
                        'completists', 'detachment', 'Tense', 'sweaty-palmed', 'Nobody', 'cared', 'grayish', 'Tufano',
                        'widescreen', 'Grabowsky', 'Coal', 'stockings', 'Utter', 'mush', 'conceited', 'pap', 'Drug',
                        'too-conscientious', 'anyplace', 'Waltz', 'strategy', 'reside', 'Lame', 'self-mutilation',
                        'pitted', 'shimmering', 'ethereal', 'winged', 'assailants', 'Kazan', 'Reversal', 'Fortune',
                        'beachcombing', 'verismo', 'indistinct', 'Ambrose', 'nurtured', 'non-porn', 'guilt-trip',
                        'Saddam', 'Hussein', 'U.N.', 'permission', 'Laggard', 'wending', 'acknowledging', 'whence',
                        'Bazadona', 'Woodard', 'relied', 'surround', 'Frankie', 'hailed', 'Pool', 'Cassel', 'vocalized',
                        'proficiency', 'colorfully', 'kitschy', 'heedless', 'impetuousness', 'embellishment', 'doltish',
                        'skunk', 'odor', 'talking-animal', 'kilted', 'out-of-kilter', 'rambles', 'self-pitying',
                        'endgame', 'moron', 'one-sided', 'over-indulgent', 'tirade', 'self-glorified', 'lovefest',
                        'Hopelessly', 'bombshell', 'temporal', 'burden', 'larded', 'sands', 'tightly', 'ordered',
                        'Doyle', 'Hmmm', 'stuffing', 'electric', 'pencil', 'sharpener', 'shattering', 'silent-movie',
                        'agreeably', 'disagree', 'techno-tripe', 'quintet', '112-minute', 'wavers', 'cellular',
                        'Sullivan', 'fuddled', 'classicism', 'exasperated', 'noticeable', 'ballet', 'categorize',
                        'smutty', 'Cows', 'Retard', 'Adroit', 'galvanize', 'Campion', 'giants', 'metal', 'fireballs',
                        'masochism', 'unencouraging', 'threefold', 'expansion', 'accompanying', 'stunt-hungry',
                        'dimwits', 'collected', 'pranks', 'spaceship', 'launching', 'pad', 'duly', 'astronauts',
                        'cabins', 'exiled', 'aristocracy', 'victorious', 'revolutionaries', 'hang', 'prior', 'bulk',
                        'centers', 'contributed', 'Niccol', 'insinuating', 'piles', 'peculiarly', 'amorality',
                        'literarily', 'chatter', 'parrots', 'Oprah', 'Simultaneously', 'heart-breaking', 'Ah-nuld',
                        'contemptuous', 'watches', 'float', 'monologue', 'well-balanced', 'Five', 'clichÃ©-laden',
                        'watered', 'co', 'AMC', 'Adobo', 'palate', 'calculations', 'slot', 'shopping', 'coordinated',
                        'yielded', 'drumbeat', 'authenticity', 'pornographic', 'swank', 'apartments', 'parties',
                        'candles', 'fret', 'calories', 'oblivious', 'overstylized', 'purÃ©ed', 'mÃ©lange', 'DePalma',
                        'drinking', 'twelve', 'beers', 'lately', 'treasured', 'variations', 'stinging', 'Baio',
                        'curtain', 'separates', 'profundities', 'large-screen', 'scooping', 'Aurelie', 'Christelle',
                        'nightmarish', 'fairytale', 'Appropriately', 'ignites', 'manically', 'lampoon', 'megaplexes',
                        'lackadaisical', 'tier', 'heartily', 'war-weary', 'marine', 'Rogers', 'shut', 'sexes', 'Topics',
                        'sailor', 'blush', 'broader', 'concrete', 'overpowered', 'VeretÃ©', 'whip-smart', 'bluffs',
                        'Synthetic', 'description', 'high-powered', 'Reassuring', 'uplifter', 'Sontag', 'Stern',
                        'Directors', 'Brett', 'Morgen', 'Nanette', 'eye-boggling', 'robotically', 'italicized',
                        'truth-in-advertising', 'hounds', 'hustling', 'Ryder', 'aided', 'Mirren', 'proficiently',
                        'trounce', '!?', 'precedent', 'fun-for-fun', 'Monte', 'Cristo', 'well-wrought', 'omits',
                        'needless', 'swordfights', 'exhibit', 'digest', 'haphazardness', 'hews', 'handed', 'gods',
                        '75-minute', 'sample', 'rubbish', 'lika', 'da', 'whirlwind', 'cringe', '295', 'indignant',
                        'Ambitious', 'psychodrama', 'rough-around-the-edges', 'Allegiance', 'vexing', 'handicap',
                        'geeked', 'archival', 'prints', 'Coy', 'Ving', 'Rhames', 'mourning', 'spiritualism', 'Ireland',
                        'singularly', 'eventful', 'guru', 'launch', 'ancillary', 'products', 'pastel', 'Brilliantly',
                        'Asphalt', 'Aloof', 'biscuit', 'consequence', 'Baby-faced', 'countless', 'dolls', 'perch',
                        'metropolitan', 'Spectacular', 'don', 'Orc', 'Uruk-Hai', 'Laugh-out-loud', 'Germany',
                        'democratic', 'Weimar', 'Republic', 'strikingly', 'devious', 'skittish', 'middle-agers',
                        'copious', 'hints', 'myriad', 'harvesting', 'redeemed', 'lazier', 'propelled', 'B-12',
                        'shimmers', 'McCracken', 'chasing', 'Ravel', 'crescendo', 'encompasses', 'paths', 'chronicles',
                        'captors', 'befuddled', 'captives', 'better-focused', 'goth-vampire', 'woe-is-me', 'represent',
                        'exemplify', 'laziest', 'bug-eye', 'dead-eye', 'tasteless', 'clout', 'doc', 'Errol', 'Morris',
                        'focusing', 'eccentricity', 'oddballs', 'hobbled', 'Bound', 'howlingly', 'triumphs', 'WASP',
                        'matron', 'Keener', 'sidekicks', 'much-needed', 'Munch', 'tenderly', 'no-holds-barred',
                        'Overcomes', 'hideousness', 'alienated', 're-invents', 'spiffing', 'leftovers', 'spicy', 'Elie',
                        'co-wrote', 'subjugate', 'tear-jerking', 'eroded', 'Aaliyah', 'governmental', 'splatter',
                        'Bratt', 'MapQuest', 'emailed', 'point-to-point', 'driving', 'bludgeoning', 'projection',
                        'immaculately', 'Patch', 'freaking', 'flippant', 'puppets', 'Kitschy', 'Sparkling', 'Travis',
                        'Bickle', 'Romething', 'ricture', 'plateau', 'interrogation', 'pours', 'twin', 'Donald',
                        'Jonze', 'Philosophically', 'logistically', 'Finch', 'nerve-raked', 'stagings', 'hardware',
                        'Baird', 'editor', 'mind-bending', 'qatsi', 'beloved', 'cardiac', 'uncluttered', 'relays',
                        'earmarks', 'gone-to-seed', 'astoundingly', 'byways', 'jabs', 'egocentricities', 'breed',
                        'Significantly', 'loser', 'Variety', 'lioness', 'protecting', 'cub', 'controlling',
                        'cultivated', 'ghoulish', 'Oversexed', 'forty', 'crushing', 'Lazily', 'Cirulnick', 'novelist',
                        'Thulani', 'Knows', 'indifferent', 'Brady', 'prosaic', 'yours', 'wince', 'facts', 'treated',
                        'morsels', 'JoÃ£o', 'Pedro', 'unwillingness', 'stench', 'juxtapositions', 'Otto-Sallies',
                        'scorcher', 'self-reflection', 'milking', 'played-out', 'dress', 'accentuating', 'muting',
                        'betters', 'Beijing', 'Traditional', 'Modern', 'overwhelms', 'fogging', 'triangles', 'Auteil',
                        'ripening', 'extreme-sports', 'creaky', 'self-empowering', 'big-wave', 'Scarcely', 'reporting',
                        'tumbleweeds', 'blowing', 'graced', 'Steinis', 'endured', 'racing', 'prologue',
                        'ill-considered', 'D.W.', 'Griffith', 'silent', 'seventy-minute', 'personally', '26',
                        'out-to-change-the-world', 'aggressiveness', 'tumultuous', 'Musset', 'milieu', 'Lightweight',
                        'coherence', 'guiltless', 'slickest', 'courtship', 'Medem', 'disrobed', 'exposed', 'guarded',
                        'virgin', 'chastity', 'tedium', 'unamusing', 'reckless', 'squandering', 'chimps', 'Goodall',
                        'minded', 'Sensual', 'electrocute', 'dismember', 'peerlessly', 'paean', 'italics', 'restless',
                        'undo', 'Beavis', 'Butthead', 'Debrauwer', 'rigidly', 'paradigm', 'permitting', 'well-worn',
                        'believed', 'small-budget', 'Picpus', 'spy-thriller', 'stroked', 'coloured', 'uni-dimensional',
                        'eyeballs', 'evaporates', 'crypt', 'mist', 'show-biz', 'subcultures', 'Merchant-Ivory',
                        'systematically', 'dear', 'split', 'retold', 'rackets', '94-minute', 'unparalleled', 'mistaken',
                        'Roses', 'trailer-trash', 'adorns', 'family-film', 'maturity', 'sneeze', 'bitten', 'drunken',
                        'scheming', 'horses', 'inadvertent', 'Claire', 'self-hating', 'lack-of-attention', 'seventeen',
                        'Kilmer', 'Doors', 'linear', 'makeup-deep', 'efficiency', 'D', 'espite', 'Daily', 'usurp',
                        'chitchat', 'neurotics', 'hypermasculine', 'implausibility', 'sags', 'courageousness',
                        'tenacious', 'appropriately', 'concludes', 'dawn', 'expand', 'lewd', 'downs', 'Wedge', 'Berg',
                        'Ackerman', 'Looney', 'Tunes', 'remade', 'diapers', '1987', 'asset', 'moon', 'fuel', 'stranded',
                        'appetites', 'spies', 'Eventually', 'peas', 'wishful', 'reinforcement', 'progresses',
                        'monotony', 'pledge', 'allegiance', 'Lacey', 'addresses', 'banter-filled', 'airy', 'bon',
                        'bons', 'accomplishments', 'Yorkers', 'Thinking', 'puppies', 'legs', 'butterflies', 'Busy',
                        'forte', 'unremittingly', 'sturdiest', 'cheapened', 'well-thought', '10,000', 'flck',
                        'assembles', 'communicate', 'undergo', 'halftime', 'fifteen', 'abiding', 'hallucinogenic',
                        'acres', 'haute', 'couture', 'conceal', 'Jeong-Hyang', 'Compared', 'flops', 'RunTelDat',
                        'fumbled', 'Ayres', 'fluidity', 'confession', 'McCrudden', 'beliefs', 'prejudices', 'galled',
                        'puddle', 'reclaiming', 'recreating', 'idiom', 'uncharted', 'sub-sophomoric', 'favorably',
                        'Das', 'Boot', 'Judging', 'eternally', 'devoted', 'insanity', 'areas', 'flinching',
                        'FearDotCom', 'Clive', 'Barker', 'lurks', 'mindset', 'flashback', 'workings', 'Bear', 'hairier',
                        'Einstein', 'unsalvageability', 'saddled', 'Keenly', 'promenade', 'Myrtle', 'S.C.', 'diner',
                        'extra-dry', '83', 'Touches', 'wistfully', 'friendly', 'skit-com', 'deposited', 'pent',
                        'ENOUGH', 'transition', 'fairness', 'varying', 'coughed', 'fidgeted', 'romped', 'aisles',
                        'bathroom', 'lector', 'thereafter', 'teeny-bopper', '60-second', 'bode', 'fisher', 'hooks',
                        'Featherweight', 'evenings', 'Efteriades', 'hug', 'squashed', 'preferably', 'semi', 'Brash',
                        'perplexing', 'Austrian', 'suppression', 'owe', 'Executed', 'hardhearted', 'Troubling', 'lags',
                        'lurches', 'not-very-funny', 'dramatics', 'resulted', 'Inventive', 'intoxicatingly',
                        'maddening', 'Performances', 'Oscar-caliber', 'Impresses', 'Manhunter', 'investment',
                        'confidently', 'orchestrated', 'aesthetically', 'rightly', 'cutoffs', 'Hush', 'low-brow',
                        'skipping', 'evoked', 'lukewarm', 'marketing', 'Breillat', 'Rain', 'sleaziness', 'steadily',
                        'murder-on-campus', 'yawner', 'anomaly', 'clotted', '105', 'locale', 'slide', 'slope',
                        'dishonesty', 'virtue', 'Hermocrates', 'Leontine', 'amours', 'Horrible', 'implied', 'PokÃ©mon',
                        'locusts', 'horde', 'hard-to-swallow', 'distinguishes', 'distinguishing', 'Awards',
                        'date-night', 'shamefully', 'strolls', 'kilt', 'shoulder', 'pre-credit', 'wounded', 'wounding',
                        'Laced', 'stable-full', 'bid', 'treachery', 'Buscemi', 'Rosario', 'witlessness',
                        'not-so-bright', 'daughter', 'trio', 'criminals', 'Fast-paced', 'thorough', 'patchwork',
                        'Begley', 'Absurdities', 'accumulate', 'lint', 'navel', 'riled', 'comforting', 'Tim', 'McCann',
                        'cockeyed', 'Insanely', 'luminous', 'addictive', 'ambivalence', 'Smokers', 'shoot-outs',
                        'fistfights', 'phlegmatic', 'reassures', 'classroom', 'concerning', 'chronically',
                        'overachieving', 'absent', 'kills', 'no-surprise', 'Aan', 'soured', 'bumps', 'icky', 'scored',
                        'powered', 'endeavour', 'mimetic', 'approximation', 'Contempt', '8Â\xa01\\/2', 'self-mocking',
                        'mock', 'power-lunchers', 'classify', 'ye', 'aircraft', 'carrier', 'sentimentalized',
                        'Butterfingered', 'big-fisted', 'Jez', 'Butterworth', 'smallest', 'sensitivities', 'clamorous',
                        'April', 'instalment', 'Independence', 'bushels', 'poky', 'pseudo-serious', 'workshops',
                        'audience-pleaser', 'doctor', 'emergency', 'hospital', 'bed', 'flowery', 'roads',
                        'Characterisation', 'sacrificed', 'Others', 'attuned', 'anarchist', 'maxim', 'interpersonal',
                        'dances', 'Steadfastly', 'uncinematic', 'soapy', 'Apple', 'Samira', 'Makhmalbaf', 'conspicuous',
                        'avid', 'impatient', 'o', 'folds', 'pseudo-bio', 'doles', 'PB', 'J', 'sandwich', 'eclipses',
                        'inertia', 'nubile', 'depravity', 'amid', 'Formulaic', '51st', 'bilingual', 'Eerily', 'cello',
                        'culled', 'funeral', 'skilled', 'Sy', 'queasy', 'infatuation', 'self-expression', 'Deserves',
                        'Nanook', 'Bledel', 'pre-teen', 'elect', 'imposed', 'appetizer', 'reeked', 'been-there',
                        'done-that', 'sameness', 'murk', 'Adrien', 'Brody', 'jump-in-your-seat', 'backseat', 'burnt',
                        'marathons', 'Pixar', 'Toes', 'great-grandson', 'thesis', 'loyalty', 'reincarnation', 'Gayton',
                        'telegraphs', 'convenience', 'Charming', 'Alien', 'Alcatraz', 'springs', 'boilerplate',
                        'ringside', 'tough-man', 'freshman', 'slump', "'53", 'unclean', 'druggy', 'spring-break',
                        'majors', 'water-born', 'Hennings', 'ensuing', 'merchandised-to-the-max', 'recruiting',
                        'playlist', 'costuming', 'disconnection', 'inextricably', 'entwined', 'head-turner',
                        'thoughtfully', 'Simple', 'affirms', 'companionship', 'counterproductive', 'uninitiated',
                        'sedate', 'Content', 'lionize', 'health', 'deep-seated', 'wildest', 'fighter', 'Batman',
                        'indecipherable', 'Deserving', 'backlash', 'Reigen', 'shrug', 'annoyance', 'chatty', 'records',
                        'warm-milk', 'conscience', 'bad-movie', 'tightened', 'bumper', 'strutting', 'Tartakovsky',
                        'freakish', 'Vaguely', 'screed', 'Epps', 'affability', 'Weissman', 'Weber', 'craziness',
                        'well-formed', '75', 'Columbia', 'Pictures', 'hard-partying', 'Serry', 'contained', 'saddest',
                        'Delhi', 'specialized', 'sharply', 'Teens', 'jaw', 'cousin', 'bottom-feeder', 'Escape',
                        'digital-effects-heavy', 'family-friendly', 'Connoisseurs', 'meticulous', 'withered',
                        'enforced', 'hiatus', 'Debut', 'grandkids', 'grandparents', 'catalytic', 'holy', 'cutthroat',
                        'annex', 'lungs', 'engages', 'actorly', 'boundary-hopping', 'innovations', 'delighted',
                        'copout', 'Pair', 'Rick', 'African-American', 'Kudos', 'trickery', 'overkill', 'tartly',
                        'doctorate', 'blustery', 'disturbingly', 'male-ridden', 'blockage', 'accompanies', 'omission',
                        'zombie-land', 'digressions', 'water-camera', 'operating', 'SchÃ¼tte', 'snapshot', 'Brecht',
                        'drew', 'abundantly', 'Cusack', 'cherished', 'show-stoppingly', 'scathingly', 'inexcusable',
                        'prisoners', 'valentine', 'Ally', 'McBeal-style', 'nine-year-old', 'Slack', 'peopled',
                        'slapping', 'foot', 'bona', 'fide', 'groaners', 'psychedelia', 'failings', 'Spanish',
                        'ambiguities', 'interdependence', 'accomodates', 'practical', 'borderline', 'dip', 'swipe',
                        'recompense', 'repetitious', 'insignificant', 'toss-up', 'presiding', 'gluing',
                        'artsploitation', 'Films', 'addiction', 'junk-calorie', 'perversely', 'mournful', 'good-time',
                        'unguarded', 'Naqoyqatsi', 'Schneidermeister', 'Makin', 'Losin', 'miniseries', 'surviving',
                        'invaders', 'existent', 'anti-virus', 'advancing', 'persistent', 'woozy', 'personified',
                        'wretched', 'dodges', 'screenful', 'gamesmanship', 'Coburn', 'inconceivable', 'Perfectly',
                        'someplace', 'self-absorption', 'insistently', 'otherness', 'telescope', 'vacuum', 'Stiff',
                        'enhanced', 'surplus', 'archive', 'Tara', 'Reid', 'six-time', 'Tropic', 'Pageant', 'aground',
                        'snared', 'slopped', 'em', 'negated', 'hypocrisy', 'kid-vid', 'lashing', 'copycat', 'defuses',
                        'submerging', 'Leaping', 'Songs', 'Floor', 'rant', 'thumbing', 'formalism', 'shape-shifting',
                        'perils', 'brushes', 'calamity', 'Majid', 'shoe-loving', 'vary', 'deafening', 'Department',
                        'telegrams', 'unwavering', 'Gallic', 'fusty', 'Viterelli', 'right-hand', 'goombah', 'Things',
                        'portent', 'unattractive', 'odorous', 'Live-style', 'Blaxploitation', 'dishes', 'ton',
                        'unprovoked', 'suffocate', 'illumination', 'sparse', 'instances', 'gloomy', 'veil', '1957',
                        'Hoult', 'scared', 'naught', 'illuminates', 'Wishy-washy', 'dismiss', 'mordant', 'floundering',
                        'Art-house', 'genre-curling', 'revives', 'free-wheeling', 'stamina', 'all-enveloping',
                        'rhapsodizes', 'Pap', 'invested', 'undergraduate', 'doubling', 'subtexts', 'stabs', 'wizard',
                        'hail', 'Sascha', 'dominant', 'Christine', 'Sylvie', 'icily', 'Les', 'Vampires', 'sparked',
                        '50s', 'humbuggery', 'hypnotically', 'Grenier', 'Heidegger', 'Nietzsche-referencing',
                        'predominantly', 'exhaustion', 'Painful', 'Felinni', 'freakshow', 'transported', 'Wladyslaw',
                        'Szpilman', 'pianist', 'Jacobi', 'writings', 'dullingly', 'Lead', 'GaÃ¯', 'limbs',
                        'conspiratorial', 'towering', 'siren', 'Old-fashioned', 'blossom', 'standardized', 'infinitely',
                        'wittier', 'Otherwise', 'marking', 'Wewannour', 'communicates', 'dumplings', 'contradiction',
                        'flexible', 'four-year-old', 'exaggeration', 'recount', 'Haunted', 'action\\/thriller',
                        'Jackal', 'Connection', 'Heat', 'pseudo-philosophic', 'nauseating', 'spinning', 'underutilized',
                        'tickled', 'humor-seeking', 'benchmark', 'lameness', 'proposes', '22-year-old', 'girlfriend',
                        '18-year-old', 'mistress', 'queasy-stomached', 'critic', 'staggered', 'blacked', 'studios',
                        'girl-on-girl', 'Enriched', 'leniency', 'deranged', 'ennui-hobbled', 'entree', 'menu',
                        'Consummate', 'actress-producer', 'Congeniality', 'Belongs', 'too-hot-for-TV',
                        'direct-to-video\\/DVD', 'one-star', 'eighth-grader', 'Runteldat', 'Trailer', 'uncool',
                        'Gadzooks', 'Lau', 'intolerant', 'toughest', 'geniality', 'fanatical', 'adherents', 'favored',
                        'moaning', 'conspicuously', 'blowout', 'Coral', 'Reef', 'Adventure', 'heavyweight', 'behalf',
                        'endangered', 'silences', 'pulchritude', 'acidity', 'bathos', "'til", 'Blanchett', 'Ribisi',
                        'weakly', 'mojo', 'get-go', 'Argentinean', 'dramedy', 'activities', 'Gently', 'smell', 'grease',
                        'Satisfyingly', 'contentedly', 'labours', 'werewolf', 'avoiding', 'leafing', 'accompanied',
                        'sketchiest', 'captions', 'Meow', 'symptom', 'counterculture', 'infrequently', 'laundry',
                        'plastered', 'precocious', 'straddles', 'fence', 'campaign-trail', 'candidate', 'reflect',
                        'plod', 'cocky', 'pseudo-intellectual', 'poetics', 'intersect', 'Suffice', 'unfulfilled',
                        'Bruckheimeresque', 'emulates', 'tainted', 'improbability', 'refracting', 'tenth', 'dicey',
                        'quirkily', 'barbed', 'poo-poo', 'cheats', 'retreats', 'penned', 'plague', 'devotedly',
                        'pin-like', 'recreation', 'splendour', 'K.', 'organized', 'strangest', 'thoughts', 'voyage',
                        'falcon', 'roughly', 'cat', 'meow', 'purr', 'forefront', 'Sixth', 'Amidst', 'sympathizing',
                        'activate', 'girlish', 'ducts', 'skinny', 'sucking', 'warmest', 'infantilized', 'populist',
                        'importance', 'hard-pressed', 'fatter', 'atypically', 'fast-edit', 'hopped-up', 'Pure',
                        'intoxication', 'uncovers', 'trail', 'craven', 'concealment', 'Rehearsals', 'Elvira', 'divorce',
                        'earthly', 'stoop', 'fateful', 'bewitched', 'adolescents', '91-minute', 'three-ring', 'surest',
                        'bet', 'all-around', 'transparently', 'foundering', 'aggravating', 'reference', 'hipness',
                        'Scarface', 'Carlito', 'fustily', 'Strangely', 'kingdom', 'deliberative', 'characterized',
                        'surface-obsession', 'typifies', 'delirium', 'pre', 'extant', 'month', 'rainbows', 'plant',
                        'smile-button', 'populace', 'Exhibits', 'salacious', 'telenovela', 'tabloid', 'embalmed',
                        'surfacey', 'singles', 'blender', 'overflows', 'unfaithful', 'happenstance', 'overladen',
                        'blight', 'horrifyingly', 'revenge-of-the-nerds', 'dredge', 'wanton', 'slipperiness', 'Corpus',
                        'reshaping', 'introduce', 'charitable', 'castrated', 'Highlander', 'mail', 'irrigates',
                        'swimfan', 'Disney-style', 'wing', 'prayer', 'hunky', 'castle', 'sky', 'Year', 'prone',
                        'indignation', 'susceptible', 'procedural', 'wanders', 'Tiger', 'Flatman', 'huge-screen',
                        'educates', 'Pascale', 'Bailly', 'rom-com', 'AmÃ©lie', 'Audrey', 'Tautou', 'fabuleux', 'destin',
                        'light-heartedness', 'weaned', 'di', 'Napoli', 'Melville', 'quick-witted', 'Lit', 'elicited',
                        'Phonce', '19th-Century', 'raindrop', 'Peppering', 'folktales', 'Villeneuve', 'Demons',
                        'warned', 'SC2', 'autopilot', 'masters', 'Blockbuster', 'fees', 'monument', 'Weird', 'Coarse',
                        'trifling', 'opposites', 'poorly-constructed', 'fallible', 'delineate', 'urges', 'Photographer',
                        'empathizes', 'bonding', 'gurus', 'doshas', 'verging', 'mumbo-jumbo', 'streaks',
                        'unimpeachable', 'unpaid', 'intern', 'typed', 'Univac-like', 'Lo', 'pie', 'graduated',
                        'Handled', 'correctly', 'soothe', 'teen-sleaze', 'depress', 'encomia', 'fruitful', 'epitaph',
                        'Exactly', 'lusty', 'documentarians', 'gooeyness', 'revisionism', 'wallflower', 'Well-made',
                        'mush-hearted', 'entering', 'synagogue', 'temple', 'jokester', 'highway', 'patrolmen', 'pains',
                        'cancer', 'grain', 'densest', 'distillation', 'Scoob', 'Shag', 'Stuffed', 'instigator', 'modus',
                        'operandi', 'crucifixion', 'juxtaposition', 'veered', 'Exxon', 'goodies', 'lumps', 'clinically',
                        'slim', 'flatula', 'advises', 'Denlopp', 'bubbly', 'exchange', 'deckhand', 'updatings',
                        'parrot', 'Morph', 'mimics', 'Excellent', 'Jacqueline', 'Plimpton', 'commitment', 'Reginald',
                        'slurs', 'Touched', 'sappiness', 'resents', 'inhale', 'gutter', 'romancer', 'Seldhal',
                        'recesses', 'unearth', 'quaking', 'Wickedly', 'consume', 'revelled', 'shallows',
                        'kid-empowerment', 'assess', 'engineering', 'Watch', 'Stupid', 'Dramatically', 'Quietly',
                        'Flamboyant', '65-year-old', '12th', 'choosing', 'actuary', 'UK', 'Carter', 'Newcastle',
                        'drips', 'tall', 'Carlin', 'T-shirt', 'sketches', 'identification', 'Fame', 'letter',
                        'narrated', 'Landau', 'lock', 'fighters', 'leaping', 'slivers', 'traits', 'gates', 'plot-lines',
                        "'80s", 'August', 'humidity', 'oh-those-wacky-Brits', 'ushered', 'correctness', 'sleepless',
                        'effectiveness', 'shoestring', 'unevenly', 'Lynch-like', 'rotting', 'be-all-end-all',
                        'modern-office', 'wanna', 'clown', 'violently', 'gang-raped', 'rancid', 'Disreputable',
                        'dampened', 'substandard', 'gonna', 'disintegrates', 'print', 'unaffected', 'greaseballs',
                        'slash-fest', 'DreamWorks', 'princesses', 'happily', 'pissed', 'summary', 'awfulness', 'pawn',
                        'disabilities', 'crassness', 'matched', 'Sodden', 'unsuccessfully', 'fuse', 'skateboard', 'arm',
                        'invited', 'residences', 'fallibility', 'third-person', 'microscope', 'rape-payback', 'hinges',
                        'enabling', 'industrial-model', 'freezers', 'serious-minded', 'concerns', 'year-end',
                        'Sonnenfeld', 'wrought', 'arguably', 'versatile', 'assign', 'escapade', 'Sofia', 'tv',
                        'tardier', 'novelty', 'webcast', 'scratches', 'itch', 'Affirms', 'swingers', 'Playboy', 'thank',
                        'goodness', 'shrugging', 'acceptance', 'evolves', 'life-changing', 'Foreman', 'barking-mad',
                        'Thewlis', 'Bettany\\/McDowell', 'hard-eyed', 'Humor', 'Bai', 'Yvan', 'writes', 'conceivable',
                        'perpetrated', 'solipsistic', 'Invigorating', 'rainbow', 'too-spectacular', 'coastal',
                        'distracts', 'good-naturedly', 'laden', 'tinged', 'undertones', 'Perry', 'shrapnel',
                        'shellshock', 'linger', 'per', 'dollar', 'hijinks', 'Lear', 'tryingly', 'Reminiscent',
                        'Jae-eun', 'Jeong', 'flowering', 'cloudy', 'becalmed', 'molehill', 'Among', 'perceptiveness',
                        'spanning', 'Marina', 'tick', 'imperfect', 'love-hate', 'Squandering', 'thoroughfare', 'native',
                        'bio-pic', 'stupefying', 'slickly', 'grey', 'antiseptic', '9\\/11', 'carol', 'pot', 'roundelay',
                        'Gator-bashing', 'Sharp', 'O2-tank', 'deportment', 'Lothario', 'quibble', 'cackles', 'Quinn',
                        'grunge-pirate', 'hairdo', 'Gandalf', 'wind-tunnel', 'cor-blimey-luv-a-duck', 'cockney',
                        'amassed', 'under-10', 'bibbidy-bobbidi-bland', 'last-place', 'basketball', 'teams',
                        'automatic', 'gunfire', 'capitalism', 'putters', 'good-naturedness', 'than-likely',
                        'celebrities', 'Making', 'trivializing', 'affirm', 'millennium', 'thoughtlessly', 'integrates',
                        'pasta-fagioli', 'savory', 'underventilated', 'pÃ¨re-fils', 'sensationalize', 'follow-up',
                        'plate', 'rapturous', 'Video', 'alternatives', 'prominent', 'counterpart', 'Set', '1986',
                        'Harlem', 'unimpressively', 'fussy', 'rounded', 'holistic', 'slimed', 'Art', 'Mora',
                        'Hitler-study', 'Snide', 'Prejudice', 'rough', 'pummel', 'playwriting', 'Victor', 'Rosa',
                        'internalized', 'Imax', 'flabbergasting', '14-year-old', 'MacNaughton', '6-year-old',
                        '10-year-old', 'wizened', 'visitor', 'faraway', 'tattoo', 'Kidnapper', 'emphasizing',
                        're-working', 'shouting', 'Lina', 'eroti-comedy', 'husband-and-wife', 'Bo', 'evasive', 'Erin',
                        'Brockovich', 'Sight', 'Eleven', 'hat-in-hand', 'stifles', 'enact', 'inter-species',
                        'Silbersteins', 'limbo', 'truth-telling', 'space-based', 'plasma', 'conduits', 'preserving',
                        'cipherlike', 'unbalanced', 'interludes', 'sorely', 'Wimps', 'off-screen', 'X-Files',
                        'dull-witted', 'disquietingly', 'Psychology', 'castles', 'scuttled', 'anatomical', 'enriched',
                        'imaginatively', 'antic', 'complexly', 'female-bonding', 'fertility', 'rivals', 'animations',
                        'clownish', 'dislocation', 'splatterfests', 'log', 'user-friendly', 'Saddled', 'unwieldy',
                        'angles', 'schedule', 'Woven', 'recalling', 'sixties', 'rockumentary', 'Lonely', 'block',
                        'unbelievably', 'Aragorn', 'Arwen', 'royals', 'scandals', 'achronological', 'blushing',
                        'gushing', 'squirts', 'Inherently', 'eccentricities', 'definitions', 'waster', 'Lux',
                        'eighties', 'grandmother', 'glides', 'reparations', 'serials', "'30s", "'40s", 'Soulless',
                        'crapulence', 'devote', 'Tyco', 'whoopee-cushion', 'IS', 'Helmer', 'dependence', 'defeats',
                        'edged', 'tome', 'Fast', 'school-age', 'serviceability', 'Food', 'tongue-tied', 'Whiffle-Ball',
                        'deploys', 'guises', 'resurrection', 'Zishe', 'fiery', 'workable', 'primer', '10th-grade',
                        'raving', 'MIBII', 'extravagantly', 'exasperatingly', 'well-behaved', 'ticks', 'accountant',
                        'Beware', 'Brit-com', 'Sunk', 'indulgence', 'scene-chewing', 'teeth-gnashing', 'actorliness',
                        'groggy', 'Possibly', 'insomnia', 'non-fan', 'Babak', 'ditty', 'Eastern', 'Fun',
                        'inside-show-biz', 'yarns', 'lucratively', 'self-caricature', 'threatened', 'acerbic',
                        'baseball-playing', 'monkey', 'skilfully', 'aerial', 'awesome', 'video-game-based', 'upfront',
                        'linearity', 'split-screen', 'stuttering', 'pompous', 'Wittgenstein', 'Kirkegaard', 'blends',
                        'uneasily', 'titillating', 'box-office', 'all-over-the-map', 'pared', 'lamer', 'saddle',
                        'Hampered', 'paralyzed', 'comprehensible', 'Dummies', 'non-techies', 'Mask', 'Blob', 'Suspend',
                        'crushed', 'fumes', 'ghosts', 'Rembrandt', 'Englishmen', 'mortality', 'affluence',
                        'underwhelming', 'even-flowing', 'extracting', 'Stallion', 'Cimarron', 'Elizabethans',
                        'tactics', 'Ado', 'amicable', 'Devoid', 'testosterone-charged', 'wizardry', 'underway',
                        'releasing', 'sending', 'organizing', 'TV-insider', 'forgot', 'rejigger', 'Twin', 'Peaks',
                        'Smoke', 'Signals', 'streamlined', 'derivativeness', 'groundbreaking', 'depictions',
                        'investigate', 'Successfully', 'laziness', 'arrogance', 'wishy-washy', 'drung', 'decisions',
                        'unsatisfactorily', 'Yourself', 'perception', 'ingeniously', 'Ends', 'Stallone', 'linking',
                        'halfwit', 'dilutes', 'stalked', 'creepy-crawly', 'bug', 'Ki-Deok', 'regular', 'bouts',
                        'defensible', 'fundamentals', 'granted', 'mishandled', 'Probably', 'Chesterton', 'jacket',
                        'noticed', 'Ethan', 'tempted', 'Gentle', 'Into', 'understatement', 'woe', 'overlook', 'goofily',
                        'well-lensed', 'gorefest', 'Q.', 'Archibald', 'lifting', 'psychic', 'uniqueness', 'squander',
                        'sincerely', 'jam', 'Incoherence', 'reigns', 'unexplored', 'Coast', 'Arliss', 'poses', 'Clue',
                        'hinge', 'threshold', 'Especially', 'genial-rogue', 'Roussillon', 'Braveheart', 'irrepressible',
                        'Ver', 'Wiel', 'Critical', 'Skillfully', 'combining', 'enlivens', 'Cardoso', 'dictates',
                        'Bittersweet', 'gestures', 'outage', 'off-kilter', 'Enormously', 'high-adrenaline', 'stuffy',
                        'give-me-an-Oscar', 'not-so-small', 'Expecting', 'humiliated', 'dogs', 'monologues', 'minority',
                        'portraiture', 'joins', 'Koshashvili', 'specificity', 'Mattel', 'executives', 'lobbyists',
                        'tinsel', 'spin-off', 'opportunism', 'Wanders', 'stylistically', 'Lynch', 'Jeunet', 'Trier',
                        'stayed', 'crowdpleaser', 'Bursting', 'from-television', 'inescapably', 'skyscraper-trapeze',
                        'Humorless', 'seriocomic', 'Georgian-Israeli', 'sixties-style', 'slickness', 'chick-flicks',
                        'treating', 'follies', 'unexamined', 'barbershop', 'chemically', 'teaming', 'Jacobson',
                        'impulses', 'Cineasts', 'in-jokes', 'Harvey', 'Weinstein', 'bluff', 'rigors', 'Denmark',
                        'Dogma', 'Vittorio', 'Sica', 'mishandle', 'physician', 'not-so-Divine', 'Re-Fried', 'Tomatoes',
                        'Dense', 'prince', 'screwy', 'method', 'imbue', 'indulges', 'Flat', 'Breitbart', 'disciplined',
                        'grade-grubbers', 'outgag', 'whippersnappers', 'Cannon', 'drumming', 'graces', 'tits', 'kiosks',
                        'pinks', 'dreamy', 'inexpressible', 'worship', 'Stress', 'participatory', 'gutsy', 'propulsive',
                        'chuckling', 'brat', 'therapy', 'Angelina', 'college-spawned', 'Colgate', 'U.', 'Broken',
                        'Cheech', 'Chong', 'CHiPs', 'Consistently', 'single-handed', 'amalgam', 'Scream', 'nationalism',
                        'media-soaked', 'rail', 'ongoing', 'deviously', 'adopts', 'guise', 'inelegant', 'unrelated',
                        'rediscovers', 'baseball', 'Maggio', 'rapes', 'pillages', 'incinerates', 'Good-looking',
                        'Clueless', 'Fork', 'eclair', 'departs', '4W', 'bill', 'funnybone', 'Bonds', 'bust', 'blondes',
                        'cross-cultural', 'hallelujah', 'gleaned', 'Refreshing', 'dynamism', 'energized', 'Volletta',
                        'Spectators', 'open-mouthed', 'wispy', 'stepmom', 'baroque', 'crummy', 'wannabe-hip', 'refers',
                        'incessantly', 'discloses', 'uselessly', 'money-grubbing', 'third-rate', 'I.Q.', 'entered',
                        'Terrible', 'fusion', 'already-shallow', 'cow', 'Asiaphiles', 'infectiously', 'Deflated',
                        'forthright', 'Zoom', 'hammering', 'upping', 'ante', 'Qualities', 'drawings',
                        'paint-by-numbers', 'gunfest', 'knucklehead', 'swill', 'just-above-average', 'nail',
                        'spirit-crushing', 'ennui', 'denuded', 'worried', 'gimmicks', 'deemed', 'hired', 'bolster',
                        'JosÃ©', 'tit-for-tat', 'retaliatory', 'responses', 'plug', 'conspirators', 'American-Russian',
                        'Armageddon', 'Girlfriends', 'wives', 'babies', 'adrenalized', 'shockers', 'Phoned-in',
                        'Talkiness', 'itinerant', 'savaged', 'dispossessed', 'Americanized', 'comparing',
                        'orchestrating', 'personable', 'jammies', 'decommissioned', 'plea', 'democracy', 'exploratory',
                        'risky', 'blueprint', 'hundreds', 'bidder', 'creatively', 'iota', 'hunk', 'paradoxically',
                        'illuminated', 'trek', 'plex', 'provocations', 'cheapening', 'sorrowful', 'niches', 'Coriat',
                        'coinage', 'reportedly', 'Swinging', 'hobby', 'attracts', '2\\/3', 'backmasking', 'Exudes',
                        'Busby', 'Berkeley', 'consumers', 'lo', 'mein', 'Tso', 'prepare', 'attach', 'steaming',
                        'cartons', 'hard-edged', 'street-smart', 'complaints', 'hard-won', 'fiascos', 'Jeopardy',
                        'fetid', 'fame', 'uglier', 'Scooter', 'Enticing', 'super-sized', 'dosage', 'omitted', 'unsaid',
                        'Balk', 'civics', 'classes', 'brainless', 'flibbertigibbet', 'valid', 'Shame', 'Vicente',
                        'Aranda', 'mad', 'rampant', 'adultery', 'style-free', 'hormonal', 'Bros.', 'Enter', 'Fist',
                        'aggressive', 'diva', 'shrewdly', 'surrounds', 'breathless', 'anticipation', 'McMullen',
                        'sugarcoated', 'natural-seeming', 'demonstrated', 'mixing', 'Noon', 'send-up', 'oozes',
                        'punishable', 'chainsaw', 'Patriot', 'Molested', 'super-serious', 'super-stupid', 'consoled',
                        'limit', 'face-to-face', 'disadvantage', 'Unspeakable', 'plaguing', 'globalizing', 'Tells',
                        'glamorous', 'thrusts', 'terrifically', 'Spielbergian', 'populates', 'glance', 'Lifetime',
                        'ample', 'large-scale', 'supplies', 'adventues', 'Terri', 'dreaded', 're-imagining', '1930s',
                        'glosses', 'Rafael', 'Light-years', 'decent-enough', 'nail-biter', 'Abysmally', 'SAT', '120',
                        'feardotcom', 'brazen', 'ambience', 'caliber', 'spinoff', 'Tattoo', 'human-scale',
                        'rip-roaring', 'beasts', 'stalking', 'hurry', 'uncompelling', 'overrides', 'vicarious',
                        'specifics', 'swims', 'Sleeper', 'Summer', 'Technically', 'Escapes', 'infusing', 'revolting',
                        'future-world', 'holographic', 'librarian', 'Orlando', 'tail', 'pay-off', 'commended',
                        'high-spirited', 'reunion', 'Berlin', 'anarchists', '1.8', 'cutting-edge', 'tighter',
                        'editorial', 'firmer', 'idling', 'Opera', 'Vera', 'instilled', 'nausea', 'conclusions',
                        'simple-minded', 'mind-numbing', 'streets', 'Bluer', 'biologically', 'beaches', 'scientific',
                        'heed', 'speeds', 'university', 'departments', 'prima', 'donna', 'Floria', 'lover', 'Mario',
                        'Cavaradossi', 'lecherous', 'Scarpia', 'unerring', 'Judith', 'Zaza', 'bedroom',
                        'self-revealing', 'hysterics', 'Dodger', 'boy-meets-girl', 'Barely', 'eternal', 'pique',
                        'disgust', 'Nadia', 'mail-order', 'bride', 'kittenish', 'accumulates', 'lax', 'meander',
                        'worn-out', 'sleep-inducingly', 'slow-paced', 'phoney-feeling', 'requiring', 'whiney', 'bugged',
                        'assurance', 'acclaim', 'insinuation', 'inadequately', 'grueling', 'time-consuming', 'ensnare',
                        'martial-arts', 'heck', 'Expect', 'discoveries', 'Directing', 'steers', 'prowess', 'Impossible',
                        'nine-tenths', 'unthinkable', 'unacceptable', 'well-put-together', 'returning', 'anecdote',
                        'seizures', 'swaying', 'cradles', 'veiling', 'Catcher', 'Rye', 'dyslexia', 'Pleasant', 'jock',
                        'adjective', 'Genuinely', 'Odd', 'Delight', 'crash', 'fillers', 'expectant', 'adoring',
                        'wide-smiling', 'reception', 'imponderably', 'statements', 'ruminations', 'true-crime',
                        'hell-jaunt', 'sip', 'wines', 'Ivory', 'iced', 'Komediant', 'hellish', "'90s", 'Well-meant',
                        'dismal', 'farcically', 'bawdy', 'regeneration', 'credulous', 'subordinate', 'morph', 'Vivi',
                        'tacked', 'thornier', 'nature\\/nurture', 'voyeur', 'Job', 'Boisterous', 'Pasolini', 'numb',
                        'eponymous', '1980', 'lingered', 'deliberateness', 'documentary-like', 'barbarism', 'cleansing',
                        'lightest', 'agitprop', 'investigation', 'characterisations', 'Vibrantly', 'hated',
                        'expressively', 'crystallize', 'minutely', 'dreamlike', 'ecstasy', 'boiled', 'argot',
                        'Rewarding', 'increase', 'imparted', 'Swiftly', 'deteriorates', 'rough-hewn', 'Andie',
                        'antagonism', 'Jerusalem', 'comeback', 'curlers', 'Canadians', 'stomps', 'hobnail', 'Natalie',
                        'Babbitt', 'chop', 'suey', 'actioners', 'schmucks', 'affords', 'Daringly', 'Biggie', 'Tupac',
                        'wobbly', 'near-future', 'amusements', 'Likely', 'grounding', 'deviant', 'behaviour', 'restate',
                        'continuing', 'landbound', 'ragged', 'ugh', 'Pratfalls', 'Alarms', 'throbbing', 'elderly',
                        'propensity', 'patting', 'greed', 'materalism', 'distressingly', 'hardest', 'wiseacre',
                        'sufficient', 'Leroy', '4\\/5ths', 'Franc', 'Desplat', 'transfixes', 'burlesque', 'howling',
                        'cringing', 'Reaches', 'losses', 'ours', 'retitle', 'Direct-to-Video', 'Awesome', 'ineffable',
                        'faint', 'Francisco', 'groupie\\/scholar', 'broadside', 'Randolph', 'illogic', 'Arnie',
                        'musclefest', 'scrutinize', 'slyly', 'anti-adult', 'Leaks', 'treacle', 'histrionic', 'eludes',
                        'Rugrats', 'respecting', 'by-the-book', 'stylings', 'Stockwell', 'grouchy', 'ayatollah',
                        'mosque', 'seedy', 'gaudy', 'shirt', 'criticizes', 'watered-down', 'sweet-tempered', 'forgoes',
                        'melange', 'slapped', 'hot-blooded', 'Sidey', 'urbanity', 'Everett', 'Wildean', 'Firth',
                        'Slight', 'compliment', 'victimized', 'modernizes', 'A.E.W.', 'Mason', 'plucks', 'Allison',
                        'Lohman', 'identity-seeking', 'situates', 'musty', 'Eagle', 'carpets', 'portrayals', 'seeds',
                        'ambivalent', 'cheered', 'reassure', 'poo', 'vulgarity', 'cussing', 'neo-Nazism', 'wretchedly',
                        'tried-and-true', 'spiritless', 'ultra-loud', 'splendid-looking', 'Forgettable', 'shoes',
                        'hoofing', 'crooning', 'gushy', 'M', 'S', 'Congrats', 'spookily', 'abrupt', 'glucose',
                        'bagatelle', 'loosely-connected', 'acting-workshop', 'exercises', 'Baker', 'Boston', 'Public',
                        'Augustine', 'conquers', 'earthy', 'beer', 'Laughably', 'irredeemably', 'long-on-the-shelf',
                        'point-and-shoot', 'must-own', 'lip-reading', 'tormented', 'faceless', 'Purports', 'hustlers',
                        'well-developed', 'ambitiously', 'abused', 'autistic', 'Romeo', 'Juliet\\/West', 'Schultz',
                        'schoolgirl', 'mystic', 'Unbreakable', 'Boring', 'Orlean', '129-minute', 'Nevertheless',
                        'robustness', 'bloodletting', 'afterschool', 'recessive', 'Mary-Louise', 'shortness',
                        'disappoints', 'Fleder', 'endear', 'indoors', 'settings', 'motionless', 'livelier', 'chore',
                        'model', 'aboriginal', 'cannon', 'cream', 'Sitting', 'Sydney', 'Darling', 'Harbour', 'seater',
                        'gliding', 'banking', 'hovering', 'solemnity', 'pump', 'overworked', 'Beresford', 'terrified',
                        'irreverent', 'scotches', 'Ã©lan', 'uhhh', 'pro-wildlife', 'immersive', 'hyper-realistic',
                        'outer-space', 'Space', 'Station', 'unapologetically', "'n'", 'utilizes', 'breathing',
                        'catapulting', 'catsup', 'hurts', 'Kaige', 'English-language', 'scripted', 'modicum', 'drops',
                        'sweet-and-sour', 'gelati', 'cross-dressing', 'gimmick', 'observer', 'defensive',
                        'life-altering', 'customarily', 'jovial', 'deficit', 'flim-flam', 'Michell', 'tick-tock',
                        'yelling', 'pact', 'negative', 'pretend', 'stimulate', 'Translation', 'respites', 'Moulin',
                        'purposeful', 'overdoing', 'lipstick', 'components', 'combine', 'PC', 'stability',
                        'integrating', 'foreground', 'forest', 'confirms', 'Lynne', 'Sounding', 'physique', 'Rudd',
                        'therapy-dependent', 'flakeball', 'spouting', 'malapropisms', 'unnamed', 'substitutable',
                        'outshined', 'LL', 'sweep', 'choppiness', 'inquisitiveness', 'Truffaut', 'soporific',
                        'tiresomely', 'dire', 'Forages', 'temperamental', 'decrepit', 'freaks', 'reprieve', 'whining',
                        '7th', 'desultory', 'Dave', 'overlapping', 'beer-soaked', 'Drowning', 'jury', 'bestowed',
                        'Gordy', 'Waldo', 'Salt', 'Screenwriting', 'honoring', 'Nicely', 'Requiem', 'feed', 'Kuras',
                        'Whereas', 'star-studded', '140', 'chilled', 'oral', 'bod', 'household', 'waters',
                        'overwhelmingly', 'overshadows', 'contagious', 'gabbiest', 'giant-screen', 'bogging', 'barrage',
                        'Joshua', 'Luis', 'attendant', 'twenty-some', 'Borstal', 'Griffin', 'Co.', 'Resurrection',
                        'distinction', 'springboard', 'Parmentier', 'contemplates', 'heartland', 'rage', 'fuels',
                        'self-destructiveness', 'surgeon', 'mends', 'meticulously', 'hour-and-a-half-long', 'hands-on',
                        'Literary', 'matinee-style', 'bang-up', 'crowds', 'hence', 'chillingly', 'Pollyana', 'barf',
                        'Passion', 'laugther', 'cascade', 'hooting', 'unengaging', 'solely', 'Borrows', 'paunchy',
                        'midsection', 'undramatic', 'infuse', 'rocky', 'reconciliation', 'spawned', 'hearing',
                        'Journalistically', 'lethally', 'Morning', 'helluva', 'singer', 'preposterousness',
                        'interference', 'movie-industry', 'inclusiveness', 'disingenuous', 'Verete', 'comprise',
                        'gestalt', 'Spider-man', 'cynic', 'societal', 'bias', 'temptingly', 'bang', 'cluelessness',
                        'misplaced', 'flog', 'obligation', 'burdened', 'accessibility', 'Poetry', 'invulnerable',
                        'playoff', 'fore', 'still-inestimable', 'Narratively', 'Puportedly', 'Events', 'convolution',
                        'eye-popping', 'secondary', 'Psycho', 'claws', 'simpler', 'leaner', 'preferable', 'untrained',
                        'sitcomishly', 'slides', 'macho', 'assert', 'Battle', 'Bots', 'mild-mannered', 'bespeaks',
                        'fantastically', 'updates', 'unreachable', 'Bollywood\\/Hollywood', 'keenest', 'Bombay',
                        'Kirsten', 'Dunst', 'diplomacy', 'Buy', 'quixotic', 'visualize', 'schizophrenia',
                        'ugly-looking', 'swaggers', 'Colosseum', 'Mormon', 'annoyed', 'Crammed', 'bristles', 'Plotless',
                        'outward', 'flakiness', 'funniness', 'slowness', 'perkiness', 'self-promotion', 'Bluto',
                        'Blutarsky', 'bottomlessly', 'assaults', 'upheaval', 'bromides', 'slogans', 'preachy-keen',
                        'tub-thumpingly', 'chump', 'discerning', 'stuffiest', 'goers', 'Tomorrow', 'patiently',
                        'Parris', 'scant', 'Poetic', 'Plato', 'therefore', 'Symbolically', 'feminine', 'Jeffs',
                        'expressiveness', 'semi-amusing', 'lyrics', 'Tonight', '99-minute', 'stink', 'Wise', 'Springer',
                        'temptations', 'unleashed', 'one-night', 'ballroom', 'waltzed', 'solace', 'prey', 'boatload',
                        'freighter', 'Compassionately', 'irreconcilable', 'lesbian', 'Tackles', 'life-embracing',
                        'cheerfully', '88', '37-minute', 'Snowman', 'sub-Tarantino', 'Typical', 'animÃ©',
                        'sword-and-sorcery', '70s', 'self-determination', 'sleep-inducing', 'intrigued',
                        'unentertaining', 'middle-age', 'piss', 'b.s.', 'Screenwriter', 'Dan', 'Shawn', 'Devolves',
                        'leaning', 'badly-rendered', 'assimilated', 'guilty-pleasure', 'daytime-drama', 'Goofy',
                        'despicable', 'insatiable', 'absorbs', '3-year-olds', 'servitude', 'skirts', '50-something',
                        'lovebirds', 'siuation', 'plastic', 'knickknacks', 'neighbor', 'garage', 'sale', 'Priggish',
                        'lethargically', 'renewal', 'innate', 'flattened', 'otherworldly', 'twinkly-eyed', 'egomaniac',
                        'Simpson', 'Dreary', 'verite', 'F.', 'techniques', 'Please', 'insular', 'dedicated', 'Plunges',
                        'celebratory', 'professor', 'Kunis', 'Horton', 'flinch', 'prognosis', 'perpetual', 'Blind',
                        'Date', 'pop-up', 'complaining', 'clocks', 'frightfest', 'italicizes', 'praying', 'Curiously',
                        'Super', 'vices', '103-minute', 'mutates', 'Letterman', 'Onion', 'fertile', 'fleeting',
                        'semimusical', 'rendition', 'done-to-death', 'grounded', 'Zellweger', 'pouty-lipped', 'poof',
                        'spindly', 'ingenue', 'Everybody', 'Goliath', '1994', 'Property', 'masochistic', 'dissecting',
                        'contours', 'brimming', 'coltish', 'neurotic', 'fudged', 'gigantic', 'lunar', 'capricious',
                        'Pootie', 'Tang', 'punchlines', 'Grenoble', 'Geneva', 'mounting', 'Kraft', 'Macaroni', 'Cheese',
                        'Prophet', 'Functions', 'collaborative', 'mergers', 'downsizing', 'busts', 'comfy', 'TUCK',
                        'EVERLASTING', 'conclusive', 'fanatics', 'hourlong', 'happily-ever', 'spangle', 'smaller',
                        'pondering', 'astronaut', 'neurosis', 'negativity', 'LDS', 'Church', 'undemanding', 'armchair',
                        'tourists', 'keg', 'Fortunately', 'sporting', 'hole-ridden', 'hard-to-predict', 'down-to-earth',
                        'nonchalant', 'meshes', 'Moving', 'drain', 'ruse', 'tactic', 'Bon', 'appÃ©tit', 'Trey',
                        'molestation', 'social\\/economic\\/urban', 'interviewees', 'ghastly', 'miscalculates',
                        'fiddle', 'garnered', 'privy', 'misconstrued', 'weakness', 'ownership', 'redefinition', 'unit',
                        'indisputably', 'hem', 'hems', 'quasi-documentary', 'Karim', 'Dridi', 'hardy', 'Theological',
                        'tongues', 'passably', 'curling', '50-million', 'US', 'gobble', 'Dolby', 'Digital', 'stereo',
                        'self-styled', 'banged', 'explain', 'appropriated', 'teen-exploitation', 'playbook', 'Intimate',
                        'grandfather', 'Egoyan', 'quandaries', 'tries-so-hard-to-be-cool', 'longest', 'lagging',
                        'trimmed', 'sprawling', 'avant', 'garde', 'Glory', 'satisfactorily', '18', 'join',
                        'monster\\/science', 'craftsmen', 'oh-so', 'homogenized', 'rose-tinted', 'glasses',
                        'marketable', '20-car', 'Kaos', 'melted', 'mesmerize', 'astonish', 'Rinzler', 'inhalant',
                        'blackout', 'true-blue', 'Singles', 'Ward', 'Celebrity', 'cinemantic', 'manual', 'coldest',
                        'crumb', '10-course', 'banquet', 'Na', 'replace', 'moviegoer', 'Delirious', 'Equlibrium',
                        'thirteen-year-old', 'Farenheit', 'Measured', 'like-themed', 'Oscar-sweeping', 'rates',
                        'Bewitched', 'Spring', 'Driver-esque', 'teetering', 'sanity', 'romanticized', 'wash', 'watched',
                        'side-by-side', 'alternate', 'half-lit', 'dorm', 'coda', 'white-knuckled',
                        'kinetically-charged', 'bucket', 'minor-league', 'libretto', 'ideally', 'jar', 'indefinitely',
                        'cattle', 'prod', 'shriveled', 'heart-felt', 'misfiring', 'atrociously', 'indescribably',
                        'Whitaker', 'Louiso', 'dawdle', 'disaffected-indie-film', 'mode', 'late-inning', 'ill-equipped',
                        'interior', 'Able', '66', 'Colorful', '10-year', 'delay', 'presses', 'limpid', 'mesmerised',
                        'crematorium', 'chimney', 'stacks', 'undermined', 'antidote', 'slather', 'Clearasil',
                        'blemishes', 'grandiosity', 'characterize', 'stricken', 'composer', 'violinist', 'overstays',
                        'Jeff', 'compositions', 'underlined', 'Finn', 'Edmund', 'McWilliams', 'implications', 'tackled',
                        'meaty', 'peppering', 'pages', 'Conforms', 're-hash', 'reworks', 'smashing', 'neglects',
                        'recharged', 'stylist', 'Interminably', 'apocalypse', 'detention', 'Stomp', 'rubber-face',
                        'Goldberg', 'island', 'P', 'artnering', 'TV-cops', 'Exceptionally', 'Sorcerer', 'prostitute',
                        'needy', 'woodland', 'Wildly', 'pessimists', 'Prime', 'calculus', 'M.I.T.', 'equations',
                        'long-winded', 'heist', 'profundity', 'Seriously', 'nonethnic', 'vivacious', 'powerhouse',
                        'enthusiasts', 'Rothman', 'angst-ridden', 'covered', 'Ordinary', '52', 'anchors', 'ballplayer',
                        'resources', 'captivatingly', 'beat-the-clock', '1960s', 'decades-spanning', 'epics',
                        'socially', 'encompassing', 'Roy', '1973', 'Sting', 'labels', 'Neverland', 'straddle',
                        'well-timed', 'rekindle', 'Adolescents', 'lascivious-minded', 'tutorial', 'surrealistic',
                        'seated', 'ignorant', 'pinheads', 'Horrendously', 'decisively', 'Friends', 'turmoil',
                        'By-the-numbers', 'D.', 'Ridley', 'twentysomething', 'hotsies', 'hard-driving', 'terminal',
                        'cutes', 'Laconic', 'frat-boy', 'bowser', '89', 'horrendously', 'dystopian', 'Fahrenheit',
                        'revived', 'anchored', 'honored', 'transform', 'mother\\/daughter', 'Andrei', 'Tarkovsky',
                        'distill', 'raucously', 'scary-funny', 'demented-funny', 'Starship', 'doorstep', 'affinity',
                        'bottom-of-the-bill', 'Chicken', 'Shankman', 'Janszen', 'bungle', 'parables', 'Downright',
                        'embarrassingly', 'reek', 'rewrite', 'garner', 'cooler', 'Woolf', 'Clarissa', 'Dalloway',
                        'frustrated', 'became', 'abhorrent', 'timelessness', 'successes', 'Mulan', 'Tarzan', 'Animated',
                        'enhance', 'self-image', 'Graphic', 'attracting', 'unfussily', 'crisper', 'punchier', 'hankies',
                        'electronic', 'Cliffhanger', 'ran', 'tonal', 'flame-like', 'trembling', 'gratitude', 'shedding',
                        'unnoticed', 'underappreciated', 'Horns', 'Halos', 'serendipity', 'overblown', 'considerably',
                        'Kubrick-meets-Spielberg', 'Delia', 'Greta', 'Paula', 'multilayered', 'Gaitskill', 'Derailed',
                        'males', 'throes', 'flush', 'testosterone', 'switches', 'gears', 'squaddie', 'intimidated',
                        'pizazz', 'sense-of-humour', 'Wrote', 'Coupling', 'disgracefully', 'scented', 'bath',
                        'post-camp', 'old-time', 'good-bad', '1972', 'benefited', 'sharper', 'Uma', 'mystique',
                        'reenacting', 'scandal', 'array', 'privileged', 'Nicolas', 'ruffle', 'truest', 'rusted-out',
                        'ruin', 'restored', 'belittle', 'paraphrase', 'fragment', 'underdone', 'potato', 'grumble',
                        'bang-bang', 'shoot-em-up', 'poster', 'Them', 'ten', 'gut-busting', 'creepy-scary',
                        're-creation', 'Croatia', 'Murray', 'prolific', 'stuffs', 'comfortably', 'Dignified', 'CEO',
                        'retreat', 'pee', 'tree', 'jokers', 'kidnappings', 'Fairly', 'Kouyate', 'elicits',
                        'governments', 'reasonable', 'tunes', 'Absolutely', 'doing-it-for', 'the-cash', 'no-bull',
                        'dynamite', 'dog-tag', 'M-16', 'whipping', 'reaffirms', 'overeager', '77', 'ticket-buyers',
                        'Show', 're-creations', 'Apart', 'best-sustained', 'Well-acted', 'well-directed', 'moodiness',
                        'bitchy', 'frolic', 'Lone', 'Steinberg', 'venality', 'pursuers', 'Kicks', 'inauspicious',
                        'mopes', 'tract', 'meanderings', 'Ignoring', 'Meeting', 'exceeding', 'Strikes', 'majestic',
                        'pro', 'Armed', 'Forster', 'Meara', 'shoots', 'namesake', 'Ludicrous', 'freak-outs',
                        'unhibited', 'hardship', 'misleading', 'unexplained', 'baboon', 'UHF', 'evanescent', 'seamless',
                        'Police', 'Decent', 'cheat', 'loopholes', 'profession', 'cinematically', 'Everlasting',
                        'everlasting', 'conundrum', 'nihilistic', 'summons', 'quadrangle', 'lamentations',
                        'self-centered', 'Cruel', 'inhuman', 'degrades', 'excruciatingly', 'blood-splattering', 'mass',
                        'drug-induced', 'evacuations', 'none-too-funny', 'distinctions', 'explosive', 'unrequited',
                        'half-hour', 'Black-and-white', 'unrealistic', 'difficulty', '8-10', 'McDonald', 'activism',
                        'Distances', 'herrings', 'register', 'fingering', 'dread', 'encountering', 'Fanboy', 'misuse',
                        'Morgan', 'Ashley', 'desecrations', 'delves', 'passive-aggressive', 'co-dependence',
                        'horror\\/action', 'Girardot', 'made-for-movie', 'evaluate', 'diary', 'Romero', 'ninth',
                        'principles', 'pointedly', 'peril', 'ruthlessly', 'pained', 'Marcus', 'copies', 'converted',
                        'McConaughey', 'irony-free', 'batting', 'limb', 'meatier', 'tipped', '+', 'humdrum', 'edit',
                        'beg', 'garish', 'diverges', 'chirpy', 'songbird', 'popped', 'Zaidan', 'zeroes', 'paycheck',
                        'fathom', 'downer', 'over-dramatic', 'omnibus', 'distressing', 'director\\/co-writer', 'mom',
                        'Doo', 'massacres', 'erupt', 'tougher', 'Kafka-inspired', 'invigorating', 'hamming',
                        'Metaphors', 'abound', 'bypass', 'Short', 'baaaaaaaaad', 'enacted', 'recently', 'forming',
                        'chain', 'circle', 'apex', 'misunderstood', 'Smokey', 'Saigon', 'simmering', 'Jose', 'Iris',
                        'double-cross', 'Heist', 'occupation', 'selves', 'desirable', 'naÃ¯vetÃ©', 'Blandly', 'Went',
                        'Ago', 'sputters', 'underdog', 'News', 'entrÃ©e', 'flavorless', 'Drama', 'temptation',
                        'salvation', 'Normally', 'sisterhood', 'character-driven', 'Moretti', 'common-man', 'Italy',
                        'beckons', 'armed', 'Oscar-nominated', 'scan', 'Lots', 'rumblings', 'tied', 'coincidence',
                        'spout', 'TV-movie', 'chapter', 'playwright', 'drinker', 'False', 'puppy', 'disagreeable',
                        'wide-eyed', 'glinting', 'Probes', 'individuals', 'spells', 'kung', 'three-minute', 'whet',
                        'appetite', 'clone', 'glow', 'impudent', 'snickers', 'humankind', 'decay', 'fourth-rate',
                        'Carrey', 'unstinting', 'qual', 'straight-up', 'yourselves', 'truths', 'teenybopper', 'Wood',
                        'pubescent', 'scandalous', 'high-strung', 'nervous', 'uncertainties', 'Jesus', 'cut-and-paste',
                        'action-movie', 'zeitgeist', 'hymn', 'Celebi', 'dared', 'produces', 'song-and-dance-man',
                        'Pasach', 'ke', 'Linklater', 'Hollywood-predictable', 'travail', 'Reagan', 'arcane', 'exposure',
                        'crassly', 'comedically', 'under-rehearsed', 'bundling', 'bouquet', 'docs', 'impacts', 'Wasabi',
                        'selling', 'Krawczyk', 'Goldie', 'hideous', 'yellow', 'exudes', 'urbane', 'bitterly',
                        'forsaken', 'P.O.V.', 'mounts', 'skateboards', 'motorcycles', 'immense', 'safely',
                        'video\\/DVD', 'babysitter', 'mini', 'car-wreck', 'Bloodwork', 'stultifyingly', 'collar',
                        'Christian-themed', 'harmed', 'insulted', 'sugar-coated', 'trombone', 'honks', 'lifted',
                        'Gilliam', 'subconscious', 'Kafka', 'casings', 'eternity', 'new-agey', 'infirmity',
                        'heart-string', 'hypertime', 'reverse', 'petty', 'thievery', 'bars', 'helpings', 'crises',
                        'ceremonies', 'Cotswolds', 'lingerie', 'models', 'dancers', 'Midwest', 'traps', 'ransom',
                        'seeping', 'neo-noir', 'governance', 'hierarchy', 'subplot', 'smuggling', 'Danish', 'cows',
                        'alter', 'ten-year-old', 'steadfast', 'dualistic', 'graze', 'consciously', 'rails',
                        'semi-autobiographical', 'Bugsy', 'caterer', 'evolve', 'Swims', 'filter', 'Disturbing',
                        'Pelosi', 'ladles', 'local', 'flavour', 'clashing', 'pizza', 'scam', 'Entertainment', 'suckers',
                        '93', 'unrecoverable', 'portraying', 'devastation', 'disease', 'dresses', 'circumstantial',
                        'tinny', 'self-righteousness', 'forewarned', 'lake', 'afterwards', 'Further', 'sense-spinning',
                        'one-trick', 'pony', 'softheaded', 'metaphysical', 'Goyer', 'unaccountable', 'precise',
                        'discordant', 'topple', 'drudgery', 'pop-music', 'movie-goers', 'embarking', 'brawn', 'testing',
                        'amiss', 'Stripped', 'tools', 'enlightenment', 'comic-book', 'Handsome', 'buffoons', 'Asia',
                        'Shu', 'insufficiently', 'findings', 'acquires', 'Freaky', 'excites', 'tickles', 'Minac',
                        'individuality', 'caricature', 'radioactive', 'odoriferous', 'crawl', 'elves', 'pimps', 'ho',
                        'one-sidedness', 'species', '90-plus', 'Ark', 'Hermitage', 'mined', 'gold', 'Ten', '13th',
                        'Clean', 'Sober', 'incarnation', 'Boat', 'Sets', 'Judaism', 'travelogue', 'Dramas', 'ferment',
                        'Gaza', 'well-shaped', 'Statham', 'hardass', 'unhidden', 'milked', 'punchline', 'oo',
                        'Arthouse', 'talkers', 'fizzability', 'Rudy', 'Lodge', 'clarify', 'trickster', 'serene', 'Lin',
                        'Chung', 'unexceptional', 'impetus', 'Returning', 'Minkoff', 'Rubin', 'Kramer', 'liven',
                        'Byzantine', 'incarnations', 'pleasuring', 'entitled', 'bow', 'fashioning', 'sure-fire',
                        'prescription', 'byplay', 'bickering', 'spy-savvy', 'siblings', 'Vega', 'Juni', 'Sabara',
                        'Cortez', 'anchor', 'give-and-take', 'dominate', 'Hanukkah', 'fried', 'Valley', 'Dolls', 'hos',
                        'splashing', 'muck', 'fillm', 'outnumber', 'three-to-one', 'publicists', 'careless',
                        'CleanFlicks', 'Ali', 'MacGraw', 'profanities', 'romance-novel', 'platitudes', 'Yakusho',
                        'long-faced', 'Shimizu', 'high-buffed', 'smartest', 'duplicate', 'Bela', 'Lugosi',
                        'now-cliched', '168-minute', 'Crummy', 'muttering', 'dissing', 'Naomi', 'petite', 'emphasising',
                        'mornings', '94', 'Smackdown', 'Amish', 'soldiers', 'sparking', 'atrocities', 'Paramount',
                        'imprint', 'non-exploitive', 'horrific', 'Stop', 'widely', 'downplaying', 'she-cute',
                        'troubled', 'homicide', 'Traffic', 'answered', 'Drop', 'derisive', 'clunker', 'fixated', 'Few',
                        'conflagration', 'shaken', 'Nesbitt', 'entries', 'celebrate', 'exceptions', 'Bullwinkle',
                        'Speed', 'minimum', 'bullfighters', 'comatose', 'ballerinas', 'bedside', 'vigils',
                        'denouements', 'genesis', 'Disneyland', 'policy', 'less-compelling', 'sugary', 'withholds',
                        'pell-mell', 'trial', 'ensures', 'draggy', 'unfurls', 'disservice', 'sneering', 'womanhood',
                        'enthusiasms', 'amble', 'rife', 'miscalculations', 'Dafoe', 'schoolboy', 'compassionately',
                        'Rabbit-Proof', 'pushiness', 'decibel', 'surrealist', 'retrospective', 'Singer\\/composer',
                        'contributes', 'slew', 'intrusive', 'exuberantly', 'serenely', 'Goddammit', 'half-assed',
                        'slummer', 'run-of-the-filth', 'recognise', 'snail-like', 'decorating', 'amok', 'childish',
                        'veering', 'Credit', 'demonic', 'conveys', 'Mexico', 'controversial', 'nonconformist',
                        'tear-stained', 'Shirley', 'Temple', 'instructs', 'lad', 'Zen', 'prickly', 'misanthropy',
                        'overinflated', 'mythology', 'recognizes', 'emphasized', 'Bicentennial', 'favour', 'cheeks',
                        'coos', 'beseechingly', 'Hutchins', 'Zelda', 'tape', 'Femme', 'Fatale', 'bait-and-switch',
                        'Turturro', 'fabulously', 'butler', 'disappearing\\/reappearing', 'concentration',
                        'Good-naturedly', 'cornball', 'hibernation', 'Hardwood', 'CuarÃ³n', 'substantive', 'structures',
                        'Altman', 'distinguishable', 'actory', 'concoctions', 'defined', 'dimness', 'fastballs',
                        'Shearer', 'snappy', 'volatile', 'movie-esque', 'pub', 'Discursive', 'ESPN', 'forms', 'doodled',
                        'Steamboat', 'Pearce', 'semi-stable', 'retooled', 'crowd-pleasing', 'grittily', 'Resources',
                        'establish', 'G.', 'fetishistic', 'well-mounted', 'ankle-deep', 'nailbiter', 'spaniel-eyed',
                        'Jean', 'slugfest', 'look-see', 'Giannini', 'Game', 'Stuffy', 'maiden', 'gore-free', 'serial',
                        'murders', 'humanize', 'snoozer', 'Tap', 'pomposity', 'tweener', 'malls', 'ou', 'cleavage',
                        'principled', 'groaner', 'zero-dimensional', 'servicable', 'flag', 'W.', 'Bush', 'laugh-free',
                        'lecture', 'Fantastic', 'crudely', 'widget', 'cranked', 'assembly', 'performing', 'ages-old',
                        'repulse', 'Olympia', 'Wash.', 'Marcken', 'Marilyn', 'reeks', 'goosebumps', 'discord',
                        'unintended', 'mother-daughter', 'grievous', 'circles', 'obsessively', 'Morrissette',
                        'lower-class', 'London', 'insouciance', 'embedded', 'simulation', 'Auschwitz', 'II-Birkenau',
                        'intrepid', 'crosses', 'exhibitionism', 'smeary', 'Irwins', 'death-defying', 'parlance', 'lick',
                        'Gedeck', 'undiminished', 'Smaller', 'kidlets', 'perfected', 'self-aggrandizing',
                        'documentary-making', 'mobility', 'wintry', '1899', 'Hatosy', 'Brendan', 'unadulterated',
                        'agonizing', 'Catch-22', 'disposible', 'guilt-suffused', 'heart-warming', 'normal', 'divisions',
                        'nonfiction', 'handy', 'Bland', 'Expands', 'depicts', 'trudge', 'rode', 'Zipper', 'corn',
                        'extra-large', 'Throws', 'ahem', 'Minutes', 'exposÃ©', 'zips', 'go-for-broke', 'heralds',
                        'Sweetly', 'fuses', 'Forrest', 'Gump', 'occurrences', 'repellantly', 'casualties', 'nada',
                        'teen-speak', 'gibberish', 'Marvelously', 'deliriously', 'Vertical', 'Limit', 'heart-pounding',
                        'flopped', 'inhabitants', 'Faces', '1938', 'appreciates', 'meager', 'perspectives',
                        'ideological', 'emaciated', 'die-hard', 'connoisseurs', 'Imperfect', 'ignoring', 'Greatest',
                        'Musicians', 'polite', 'Dante', 'styled', 'Gremlins', 'Dripping', 'bypassing', 'trivialize',
                        'Chou-Chou', 'plaintiveness', 'stiletto-stomps', 'demographics', 'unaware', 'Yang', 'Toward',
                        'morphs', 'Stacy', 'Sean', 'Penn', 'monotone', 'respects', 'foo', 'yung', 'Afternoon', 'reduce',
                        'benevolent', 'Ken', 'pivotal', 'P.O.W.', 'heartache', 'someday', 'telemarketers', 'Spare',
                        'framing', 'wholesale', 'ineptitude', 'helmer', 'neo-fascism', 'persistence', 'fiendishly',
                        'dizzy', 'disorientated', 'headline-fresh', 'orthodox', 'Bank', 'addressing', 'compromising',
                        'Superbly', 'hammer', 'swords', 'surrendering', 'composure', 'justifies', 'dependable',
                        'botched', 'inadvertently', 'sidesplitting', 'Creates', 'mythologizing', 'heroism', 'abject',
                        'overweight', 'Spinning', 'overstating', 'Total', 'Recall', 'fun-seeking', 'distractions',
                        'demeaning', 'developers', 'Commerce', 'tourism', 'pageants', 'neglecting', 'studio-produced',
                        'suitcase', 'subjective', '15-year', 'large-format', 'Del', 'anti-Catholic', 'protestors',
                        'ravaging', 'recreated', 'underlay', 'gaiety', 'Li', 'Ziyi', 'stakes', 'auto-critique',
                        'clumsiness', 'censure', 'Brussels', 'sprouts', 'horizons', 'dullness', 'samurai', 'Jiang',
                        'Wen', 'Doorstep', 'wartime', 'Heller', 'Vonnegut', 'contributions', 'Real-life', 'strongman',
                        'setup', 'cheap-shot', 'Finally', 'French-produced', 'Nice', 'unsophisticated', 'Millions',
                        'heaped', 'reap', 'Rises', 'oh-so-Hollywood', 'rejiggering', 'fearlessly', '80-minute',
                        'Control-Alt-Delete', 'gallery', 'capitalizes', 'spill', 'projector', 'lens', 'progressive',
                        'china', 'provocateur', 'special-interest', 'slaps', 'liberalism', 'Precocious',
                        'smarter-than-thou', 'propriety-obsessed', 'online', 'laptops', 'plans', 'arriving',
                        'reminders', 'bind', 'peekaboo', 'resonances', 'fringes', 'underground', 'uproar', 'MPAA',
                        'Disappointing', 'immortal', 'mountains', 'contenders', 'September', 'tiring', 'storytellers',
                        'prom', 'dates', 'Lush', 'stills', 'mattered', 'scorn', 'appearing', 'Heavy', 'rolls',
                        'machinations', 'indispensable', 'agony', 'plummets', 'esoteric', 'musings', 'polemical',
                        'self-defeatingly', 'decorous', 'niblet', 'beside', 'Fudges', 'purer', 'Highlighted', 'artless',
                        'sytle', 'tremendously', 'overplayed', 'aimlessness', 'intergalactic', 'sillier', 'Lantern',
                        'lobotomy', 'scooped', 'discarded', 'fifty', 'Somehow', 'Pryce', 'La', 'Salle', 'off-center',
                        'Fabian', 'diatribes', 'households', 'faux-urban', 'hotter-two-years-ago', 'R&B', 'Grating',
                        'bronze', 'groove', 'instrument', 'retail', 'clerk', 'wheels', 'Jiri', 'Menzel', 'Closely',
                        'Watched', 'Trains', 'Danis', 'Tanovic', 'giggle', 'lowered', 'classification', 'blips',
                        'Marvin', 'Gaye', 'Supremes', 'Sheds', 'wiser', 'shambles', 'ventures', 'abilities', 'absorb',
                        'Colin', 'expressly', 'sewage', 'shovel', 'gullets', 'simulate', 'sustenance', 'over-blown',
                        'scar', 'cold-hearted', 'tissue-thin', 'swung', 'dropped', 'Uzumaki', 'parallel', 'prostituted',
                        'decomposition', 'cloak', 'romanticization', 'delusional', 'grooved', 'raison', "d'etre",
                        'young-guns', 'grumbling', 'grub', 'stepmother', 'wild-and-woolly', 'wall-to-wall', 'payoffs',
                        'Depressingly', 'exhaustingly', 'Hitchens', 'flower', 'purge', 'claw', 'vainglorious', 'Cheap',
                        'crawls', 'snail', 'anew', 'dehumanizing', 'ego-destroying', 'unemployment', 'Falsehoods',
                        'creator', 'Shaw', 'melting', 'Phocion', 'attentions', 'passionately', 'uncover', 're-release',
                        'Ron', 'packages', 'acidic', 'all-male', 'Eve', 'Intermezzo', 'full-frontal', 'printed', 'Iles',
                        'infuriatingly', 'uncut', 'Sergio', 'Leone', 'staggering', 'Upon', 'in-between', 'champagne',
                        'girls-behaving-badly', 'underlies', 'sneers', '50-year-old', 'unslick', 'disconcertingly',
                        'marginally', 'doubtful', 'Audiences', 'squint', 'egregious', 'lip-non-synching', '72', 'Melds',
                        'Kate', 'Janine', 'all-woman', 'cafÃ©', 'overhearing', 'late-twenty-somethings', 'natter',
                        'tables', 'sugar', 'hysteria', 'PlayStation', 'abrasive', 'groan-to-guffaw', 'ratio',
                        'skirmishes', 'waged', 'predators', 'buff', 'marred', 'disastrous', 'draw', 'walls', 'shapable',
                        'bio', 'Barris', 'architect', 'Lurid', 'lucid', 'polemic', 'loathe', 'unsolved', 'unresolved',
                        'jockey', 'kindred', 'tailor-made', 'Cliff-Notes', 'full-length', 'Efficient', 'Tommy', 'clean',
                        'peep', 'booths', 'mopping', 'kindness', 'survivable', 'T.', 'Simplistic', 'honorable',
                        'Problem', 'mockumentary', 'accumulated', 'pokey', 'hackery', 'pressure', 'cooker', 'horrified',
                        'Schnieder', 'bounces', 'wrists', 'tummy', 'huggers', 'twirling', 'worlds', '170', 'sexiness',
                        'underlines', 'tangents', 'Piscopo', 'Chaykin', 'Headly', 'collect', 'synergistic', 'examining',
                        'trimming', 'expeditious', 'dumped', 'boiling', 'simmer', 'Jaglomized', 'annual', 'Riviera',
                        'spree', 'blab', 'gesturing', 'encourages', 'Pender', 'wood', 'folksy', 'binary', 'oppositions',
                        'Banal', 'sulky', 'overcoming-obstacles', 'sports-movie', 'transparency', 'Future', 'boomers',
                        'paved', 'slam-dunk', 'Cary', 'Katherine', 'insufferably', 'constricted', 'gamely', 'clean-cut',
                        'fiendish', 'psychologizing', 'bridge', 'self-knowledge', 'lab', 'fresh-squeezed',
                        'high-wattage', 'brainpower', 'coupled', 'unfakable', 'camouflaging', 'Broder', 'Ignore',
                        'methodology', 'alienate', 'ringing', 'indie-heads', 'yawn-provoking', 'farm', 'mingles',
                        'too-long', 'spoofy', 'Macbeth', 'interlocked', 'comparatively', 'repetitively', 'awash',
                        'droning', 'conscious', 'fosters', 'Playfully', 'crazier', 'skyscraper', 'nursery',
                        'preliminary', 'piecing', 'trouble-in-the-ghetto', 'teasing', 'Befuddled', '96', 'fluxing',
                        'silly-looking', 'Morlocks', 'sleight-of-hand', 'ill-wrought', 'hypothesis', 'Graffiti',
                        'inversion', 'Concubine', 'eschews', 'panorama', 'gay-niche', 'aiming',
                        'soon-to-be-forgettable', 'action-thriller\\/dark', 'crudities', 'Lagaan', 'police-oriented',
                        'participation', 'TNT', 'Original', 'Johnnie', 'Wai', 'Ka', 'Fai', 'interests', 'lying',
                        'ribcage', 'grip', 'distorts', 'riddles', 'gut-clutching', 'advocacy', 'torrent', 'dog-paddle',
                        'focuses', 'Loosely', 'Burkina', 'Faso', 'Jeffrey', 'jazz-playing', 'exterminator', 'Sunshine',
                        'digitally', 'altered', 'monitor', 'Friedman', '1970', 'stultifying', 'deleting', 'advanced',
                        'Prozac', 'Nation', 'rampantly', 'pandemonium', 'escapades', 'gander', 'amuses',
                        'Enthusiastically', 'Becker', 'hang-ups', 'fresher', 'lifelong', 'formalist', 'billing',
                        'GarcÃ\xada', 'Bernal', 'TalancÃ³n', 'cases', 'Corcuera', 'Rymer', 'conjure', 'followers',
                        'dead-undead', 'shrieky', 'well-rounded', 'achieved', 'lottery', 'Memorable', 'renders',
                        'ritual', 'even-toned', 'carousel', 'spousal', 'fling', 'monosyllabic', 'scribe', 'Wade',
                        'workmanlike', 'attentive', 'deem', 'largest-ever', 'eloquence', 'challenge', 'Testament',
                        'centuries', 'kitchen-sink', 'occurs', 'rekindles', 'muckraking', 'virulent', 'foul-natured',
                        'pics', 'el', 'margaritas', 'sexist', 'brow', 'snuck', 'lovably', 'Two-bit', 'uproarious',
                        'anticipated', 'calibre', 'Throw', 'Afghan', 'warlord', 'dustbin', 'winningly', 'Aspires',
                        'cracked', 'Buckaroo', 'Banzai', 'Fairlane', 'enthrall', 'gangster\\/crime', 'movie-specific',
                        'breaking', 'robberies', 'father-and-son', 'Competently', 'Rashomon-for-dipsticks', 'overtake',
                        'Prepare', 'Left', 'sensation', 'Le', 'CarrÃ©', 'burnt-out', 'prettiest', 'outpaces',
                        'contemporaries', 'redolent', 'augmented', 'resuscitate', 'fun-loving', 'handbag-clutching',
                        'Viva', 'le', 'Resistance', 'Implicitly', 'chicanery', 'self-delusion', 'businesses', 'dues',
                        'unread', 'journalistic', 'Tedious', 'snagged', "'n", 'milquetoast', 'Garth', 'progressed',
                        'rejects', 'patent', 'dramatize', 'S1M0NE', 'Follows', 'bleed', 'second-rate', 'Obviously',
                        'hippie-turned-yuppie', 'doe-eyed', 'pre-9', 'Homeric', 'spotlights', 'unforgivingly', 'Sarah',
                        'psychotic', 'outweighs', 'Caviezel', 'embodies', 'McTiernan', 'lighter', 'sober-minded',
                        'tap-dancing', 'rhino', 'over-familiarity', 'hit-hungry', 'strip-mined', '1997', 'tube',
                        'mystical', 'puzzled', 'iconic', 'Overwrought', 'Dogme', 'bedfellows', 'compatible',
                        'mystification', 'impressionable', 'Pompous', 'garbled', 'Mediterranean', 'Swept', 'Entertains',
                        'scrape', 'cracker', 'disoriented', 'disarming', 'Wonderland', 'stalker', 'Newfoundland',
                        'prevalent', 'curtains', 'zzzzzzzzz', 'Feel', 'Ismail', 'History', 'well-contructed', 'stepdad',
                        'adjusting', 'usher', 'tradition-bound', 'Uwe', 'blah', 'recovers', 'too-frosty', 'Paltrow',
                        'authenticate', 'slop', 'travel-agency', 'mud', 'independent-community', 'guiding', 'Top-notch',
                        'Maguire', 'dimensional', 'sunny', 'disposition', 'jealous', 'Elysian', 'Fields', 'smallness',
                        'sought', 'decipherable', 'intermingling', 'naivetÃ©', 'shackles', 'espionage', 'plumbed',
                        'deferred', 'explored', 'psychodramatics', 'double-pistoled', 'ballistic-pyrotechnic',
                        'Alternating', 'android', 'fantasized', 'maggots', 'crawling', 'Pan', 'Nalin', 'practitioners',
                        'unsung', 'boldface', 'Manipulative', 'period-piece', 'blarney', 'squeezed', 'Affable',
                        'rawness', 'reverberates', 'postapocalyptic', 'Yuen', 'McCulloch', 'disturbed', 'poise',
                        'non-narrative', 'near-impossible', 'gesture', 'Shindler', 'hollowness', 'Professional',
                        'doubts', 'yearnings', 'misunderstanding', 'Marivaux', 'adhering', 'grateful', 'bone-chilling',
                        'modem', 'disconnects', 'fullness', 'negate', 'painless', 'time-killer', 'recovering',
                        'undogmatic', 'evenly', 'milks', 'ever-watchful', 'Runs', 'Frankenstein-monster', 'Samuel',
                        'lisping', 'reptilian', 'bask', 'adrift', 'unifying', 'Jaw-droppingly', 'Ritter', 'Three',
                        'Ponderous', 'Tiresomely', 'hammily', 'collegiate', 'restoring', 'Lampoon', 'irrelevancy',
                        'Masseur', 'riding', 'Ganesh', 'Behan', 'lugubrious', '3\\/4th', 'spry', '2001',
                        'sequel-for-the-sake', 'of-a-sequel', 'critic-proof', 'hit-and-miss', 'suspecting',
                        'day-to-day', 'thrillingly', 'Patric', 'splendidly', 'Mariah', 'Carey', 'Wisegirls', 'commerce',
                        'raw-nerved', 'pig', 'flails', 'limply', 'pallid', 'archetypal', 'ART', 'FILM', 'ultra-cheesy',
                        'back-stabbing', 'inter-racial', 'importantly', 'meddles', 'argues', 'kibbitzes', 'Breen',
                        'actorish', 'notations', 'exalts', 'Marxian', 'harmoniously', 'joined', 'Tres', 'Nia',
                        'worldly-wise', 'looseness', 'grasping', 'workshop', 'Notice', 'luscious', 'Needed',
                        'bling-bling', 'throat', 'reaffirming', 'energizing', 'charting', 'turntablism', 'Taiwanese',
                        'Katz', 'Alfonso', 'rapid', 'omniscient', 'harps', 'media-constructed', 'leers',
                        'romantic\\/comedy', 'souvlaki', 'indigestion', 'Ill-considered', 'flimsier', 'out-sized',
                        'logically', 'porous', 'Devotees', 'Wrath', 'Khan', 'nagging', 'whimsicality', 'yields',
                        'reflected', 'threatens', 'dramaturgy', 'surge', 'swirling', 'rapids', 'rouses', 'trained',
                        'avalanches', 'fuzziness', 'anti-erotic', 'Bloody', 'Sunday', 'prevention', 'half-asleep',
                        'suddenly', 'accused', 'undisciplined', 'lectured', 'tech-geeks', 'domineering', 'Reader',
                        'Digest', 'Carlos', 'bubbles', 'battered', 'Straightforward', 'Driven', 'Lookin',
                        'American-style', 'Anybody', 'Shining', '1980s', 'clichÃ©-riddled', 'Ia', 'Drang', 'Personally',
                        'thrown-together', 'summer-camp', 'underrehearsed', 'arbitrarily', 'comical', 'probes',
                        'Afghani', 'streamed', 'Sleek', 'Bravado', 'arrogant', 'unimaginatively', 'foul-mouthed',
                        'Adrift', 'Bentley', 'Hudson', 'sniffle', 'respectively', 'Ledger', 'stimulus', 'optic',
                        'forgivable', 'compensate', 'old-fashioned-movie', 'unburdened', 'Chardonne', 'detractors',
                        'borrow', 'stellar', 'unhappiness', 'whoop', 'noisy', 'Belinsky', 'infants', 'pollute', 'dodge',
                        'medium-grade', 'fitfully', 'weightless', 'draft', 'churn', 'Jar-Jar', 'Binks', 'whine',
                        'bellyaching', 'homosexual', 'labyrinthine', 'buffeted', 'underdogs', 'tar', 'lawmen',
                        'Boasting', 'self-sacrifice', 'ungainly', 'knees', 'prepackaged', 'hard-sell',
                        'image-mongering', 'publicist', 'enamored', 'food-spittingly', 'Certain', 'groan', 'hiss',
                        'underworld', 'Driver', 'ingredient', 'four-hour', 'commonplace', 'seamy', 'self-satisfaction',
                        'schools', 'tomfoolery', 'Compulsively', 'Galinsky', 'Hawley', 'planned', 'documentarian',
                        'rope', 'jolted', 'gourd', 'embellished', 'enlightened', 'Manoel', 'Oliviera', 'closure',
                        'Nancy', 'Savoca', 'forged', 'still-raw', 'unsettled', '9-11', 'glass-shattering', 'Flashy',
                        'whirling', 'distract', 'introverted', 'fetishes', 'curtsy', 'One-sided', 'explanations',
                        'supposedly', 'thud', 'tendencies', 'Trade', 'Center', 'moratorium', 'treacly', 'professors',
                        'heartwarmingly', 'motivate', 'Bogdanich', 'pro-Serbian', 'Clyde', 'Barrow', 'depending',
                        'Ignorant', 'Fairies', 'indoctrinated', 'prejudice', 'rash', 'uncharismatically', 'rÃ©sumÃ©',
                        'Bar', 'irrevocable', 'agendas', 'signature', 'Disjointed', 'Pinochet', 'spoken', 'Patricio',
                        'Guzman', 'Weighted', 'Greg', 'full-fledged', 'Ear-splitting', 'crash-and-bash',
                        'lovable-loser', 'unfold', 'stripped', 'proverbial', 'overacted', 'Inconsequential',
                        'road-and-buddy', 'attending', 'Fifty', 'not-too-distant', 'analgesic', 'overstimulated',
                        'remained', 'amazement', 'participate', 'ill-advised', 'troubling', 'all-powerful', 'pop-cyber',
                        'feeds', 'Bjorkness', 'Woefully', 'L', 'ame', 'Merely', 'toys', 'paeans', 'conjuring',
                        'flashing', 'campfire', 'willies', 'Harks', 'cellophane-pop', 'Ladies', 'Gentlemen', 'Fabulous',
                        'Stains', 'strenuous', 'clinch', 'Razzie', 'superfluous', 'plagued', 'mighty', 'Guilty',
                        'attributable', 'Final', 'merited', 'hostage', 'Sleepless', 'Seattle', 'Passions', 'loneliest',
                        'virtuous', 'lending', 'Quirky', 'lumpy', 'two-day', 'paws', 'un-bear-able', 'Gidget',
                        'muscles', 'smarts', 'Confusion', 'favourite', '146', 'Cloaks', 'anti-feminist', '=',
                        'romantic-comedy', 'duds', 'eccentrics', 'shifted', 'water-bound', 'land-based', 'wreckage',
                        'Battlefield', 'Hypnotically', 'wail', 'cadence', 'fifteen-year-old', 'suicidal', 'beast',
                        'swathe', 'retaining', 'refusing', 'bullseye', 'timeout', 'affect', 'Kurds', 'Phifer', 'Cam',
                        'ron', 'reward', 'provincial', 'salt', 'back-story', 'underplays', 'Wyman', 'June', 'Cleaver',
                        'Well-done', 'parapsychological', 'phenomena', 'whir', 'clicks', 'dug', 'earplugs',
                        'pillowcases', '87', 'reviewers', 'clear-cut', 'all-out', 'Thornberry', 'wedgie', 'Nakata',
                        'overuse', 'waits', 'dusty', 'leatherbound', 'deer', 'dies', 'quietude', 'lived-in', 'churlish',
                        'begrudge', 'jaw-dropping', 'boffo', 'punches', 'led', 'infamy', 'lightness', 'tots', 'till',
                        'Texan', 'meetings', 'recycles', 'gays', 'repugnant', 'Roland', 'JoffÃ©', 'Demi', 'Scarlet',
                        'Letter', 'Polished', 'well-structured', 'lemon', 'Stylistically', 'sulking', 'powder',
                        'sun-splashed', 'whites', 'Tunis', 'belly-dancing', 'pre-WWII', 'fictional', 'causes', 'Gai',
                        'befallen', 'immune', 'camerawork', 'warmed-over', 'Imogen', 'Kimmel', 'revision', 'enhancing',
                        'morals', 'endings', 'Chasing', 'constrictive', 'Eisenhower', 'tranquil', 'teen-driven',
                        'toilet-humor', 'codswallop', 'Killing', 'Barlow', 'rappers', 'interview', 'Suge', 'Knight',
                        'romantics', 'grossest', 'secretions', 'responsibilities', 'toddler', '71', 'extrusion',
                        'handiwork', 'OK', 'Shohei', 'beginnings', 'Schweig', 'exude', 'warrior', 'millions',
                        'Potty-mouthed', 'schizo', 'capped', 'bombs', 'Hollywood-action', 'dispatching',
                        'understandable', 'motivation', 'lazily', 'conversion', 'Branagh', 'Anakin', 'lustrous',
                        'courtroom', 'wised-up', 'skidding', 'patch', '270', 'large-frame', 'scientists', 'chokes',
                        'vainly', 'Comic', 'Simpsons', 'matchmaking', 'Beginners', 'Giler', 'bags', 'Myers', 'buying',
                        'heavyweights', 'Zemeckis', 'agreed', 'cavorting', 'ladies', 'underwear', '127', 'despairing',
                        'establishment', 'let-down', 'Plus', 'mentioned', 'ransacks', 'archives', 'quick-buck',
                        'senseless', 'slips', 'enables', 'navigate', 'Faithful', 'Hatfield', 'Hicks', 'oddest',
                        'couples', 'gambles', 'ramifications', 'seldom', 'honorably', 'Kahlories', 'frustrations',
                        'sitcom-worthy', 'Otar', 'Iosseliani', 'worker', 'respite', 'refresh', 'Feeble', 'jazz',
                        'Globetrotters-Generals', 'Juwanna', 'guy-in-a-dress', 'obscenely', 'Sidewalks',
                        'snow-and-stuntwork', 'upstaged', 'avalanche', 'holiday-season', 'uninspiring', 'Episode',
                        'Attack', 'Clones', 'top-heavy', 'blazing', 'cheatfully', 'bloodsucker', 'reassuringly',
                        'aching', 'Chalk', 'hubristic', 'representing', 'cross-section', 'bounds', 'rat-a-tat',
                        'wearisome', 'saps', 'inarticulate', 'book-on-tape', 'Kid', 'Stays', 'Picture', 'abridged',
                        'interpreting', 'phantasms', 'pasts', 'IQ', 'Our', 'burrito', 'all-night', 'tequila', 'bender',
                        'jackass', 'millennial', 'undying', 'politesse', 'photographic', 'sorts', 'recognizably',
                        'Greenfingers', 'telanovela', 'traveled', 'devastated', 'famine', 'documented', 'cruelty',
                        'arch', 'cold-blooded', 'triviality', '100-minute', 'roster', 'Kathryn', 'progression', 'error',
                        'Aggravating', 'flesh-and-blood', 'Cross', 'Hungry-Man', 'portions', 'preserves', 'ardor',
                        'Affectionately', 'traced', 'Done', 'wasting', 'Legally', 'Blonde', 'Abomination',
                        'Blisteringly', 'scarily', 'sorrowfully', 'surveys', 'Holden', 'something-borrowed', 'wan',
                        'sketched', 'mixed-up', 'soul-stirring', 'Israeli\\/Palestinian', 'Veggies', 'hell-bent',
                        'imaginable', 'variant', 'nincompoop', 'Sandlerian', 'manchild', 'Trip', 'worthless',
                        'open-hearted', 'Fathers', 'bonds', 'crave', 'Accidental', 'goose-pimple', 'bruised',
                        'Peppered', 'wattage', 'late-summer', 'surfer', 'Massoud', 'cultivation', 'devotion', 'moat',
                        'moldy-oldie', 'not-nearly', 'as-nasty', 'as-it', 'thinks-it-is', 'funky', 'artificiality',
                        'mankind', 'idol', 'hooked', 'pulpiness', 'reverie', 'Mermaid', 'Aladdin', 'clouds', 'chords',
                        '1989', 'fuelled', 'sterling', 'genre-busting', 'hyped', 'claustrophic', 'attics', 'eviction',
                        'dismay', 'pleas', 'wildlife', 'environs', 'unlaughable', 'McCoist', 'goal', 'frighten',
                        'disturb', 'shiver-inducing', 'nerve-rattling', 'clams', 'broiling', 'shimmeringly', 'watery',
                        'brown', 'Get', 'pooper-scoopers', 'salient', 'smothered', 'misty-eyed', 'tear-drenched',
                        'quicksand', 'Frequent', 'flurries', 'slaloming', 'generalities', 'scrutiny', 'aristocrats',
                        'Confuses', 'contorting', 'dipped', 'preaches', 'suppose', 'dad', 'Genevieve', 'LePlouff',
                        'option', 'creators', 'Hispanic', 'teeth-clenching', 'gusto', 'Molina', 'careening', 'salvage',
                        'Zeus', 'Kangaroo', 'notches', 'pablum', 'astringent', 'certified', 'palatable', 'Proof',
                        'gathering', 'MGM', 'Splash', 'threads', 'morose', 'pregnancy', 'rape', 'Buries', 'borrowed',
                        'Chips', 'Block', 'impish', 'divertissement', 'Gainsbourg', 'Shakesperean', 'yarn-spinner',
                        'compels', 'envelope', 'plods', 'methodically', 'communicating', 'irritates', 'saddens',
                        'obnoxiously', '2,500', 'Bubba', 'Ho-Tep', 'languishing', 'Twilight', 'decisive', 'elite',
                        'proceed', 'implodes', 'fortify', 'hoopla', 'breakdown', 'buck', 'greasy', 'vidgame',
                        'freak-out', 'slanted', 's\\/m', 'deadeningly', 'drawn-out', 'Looking', 'Leonard', 'uphill',
                        'muster', 'Releasing', 'January', 'Caton-Jones', 'gleefully', 'grungy', 'receives', 'Luke',
                        'Bubble', 'Goo', 'contribute', 'Complex', 'sinuously', 'off-puttingly', 'corniest', 'Charly',
                        'imitative', 'innumerable', 'derisions', 'long-lived', 'track', 'Spectacularly', 'virtuoso',
                        'throat-singing', 'transgression', 'virtuosic', 'seller', 'Foolish', 'Choices', 'watercolor',
                        'Dumbo', 'affectionately', 'unafraid', 'stubbornly', 'refused', 'aggrieved', 'learns', 'Banger',
                        'Sisters', 'SLC', 'command', 'Mitch', 'wall', 'shudder', 'tremble', 'weighed', 'goodly',
                        'Believability', 'contemplative', 'bloodbath', 'elegy', 'Mordantly', 'intimately',
                        'actor\\/director', 'Polson', 'award-winning', 'Giles', 'Nuttgens', 'sleek', 'advert',
                        'Audacious-impossible', 'prescient', 'skateboarder', 'Tony', 'BMX', 'rider', 'Mat', 'Turks',
                        'angling', 'Intensely', 'rat', 'burger', 'CD', 'redundancies', 'inexpressive', 'backward',
                        'prefeminist', 'During', 'Antonio', 'Liu', 'fire-breathing', 'Schindler', 'grandiloquent',
                        'grotesquely', 'Continually', 'perceptions', 'stranger', 'affectingly', 'Shattering',
                        'maladjusted', 'narcotized', 'shocker', 'loyal', 'wherein', 'ne', 'sturdiness', 'solidity',
                        'Bottom-rung', 'high-end', 'Hughes', 'Elder', 'snapping', 'mordantly', 'soullessness',
                        'ascertain', 'pessimistic', 'dunno', 'undergrad', 'gotten', 'pop-influenced', 'prank',
                        'transcendence', 'Unambitious', 'animated-movie', 'walled-off', 'SONNY', 'dazzle', 'longtime',
                        'neophyte', 'crummy-looking', 'labelled', 'Sewer', 'rats', 'skeeved', 'painkillers',
                        'bewilderingly', 'Satan', 'R&D', '24\\/7', 'self-assured', 'filmgoing', 'rhapsodize',
                        'dirgelike', 'fang-baring', 'lullaby', 'preciseness', 'dominated', 'revigorates', 'sway', 'w',
                        'circa', 'Sellers', 'Greengrass', 'tad', 'intelligibility', 'Methodical', 'purposefully',
                        'reductive', 'sun-drenched', 'parlor', 'creep', 'assaultive', 'gunfight', 'Large',
                        'highlighted', 'post-war', 'instruct', 'reeking', 'library', 'welcomes', 'dash', 'monopoly',
                        'world-renowned', 'Iben', 'Hjelje', 'Includes', 'offended', 'chomp', 'ants', 'arrow',
                        'unscathed', 'raging', 'clones', 'Essentially', 'markedly', 'inactive', 'conversational',
                        'confessional', 'flashbulb', 'absence', 'bout', 'replacing', 'Besotted', 'misbegotten',
                        'openly', 'revitalize', 'clung-to', 'provokes', 'expressionistic', 'underscoring', 'slathered',
                        'heartbreakingly', 'singular', 'schticky', 'profiling', 'Latin', 'trend', 'overdue',
                        'aristocrat', 'finery', 'deflated', 'incurably', '22', 'eighth', 'Hopkins\\/Rock', 'personas',
                        'flux', 'grinds', 'Winger', '1995', 'Paris', 'writer\\/directors', 'VeggieTales', 'appetizing',
                        'asparagus', 'Times', 'BÃ¼ttner', 'intractable', 'irreversible', 'Criterion', 'discovered',
                        'indulged', 'returned', 'warn', 'producing', 'Jon', 'Purdy', 'Kathie', 'Gifford', 'knitting',
                        'needles', 'catharsis', 'homework', 'soaked', 'origins', 'Nazi', 'aesthetics', 'Memories',
                        'faded', 'skipped', 'McDormand', 'marry', 'Fisk', 'counter', 'crudity', 'Strictly',
                        'semi-surrealist', 'cornpone', 'Cosa', 'Nostra', '65-minute', 'trots', 'vacation', 'Stonehenge',
                        'Wes', 'Nightmare', 'Elm', 'Street', 'Hills', 'Eyes', 'schlock', 'merchant', 'ramblings',
                        'smorgasbord', 'soliloquies', 'boosted', 'size', 'graves', 'Hare', 'Cunningham', 'forwards',
                        'markers', 'Shunji', 'Skin', 'kids-in-peril', 'cope', 'suggesting', 'Festers', 'dungpile',
                        'monkeys', 'flinging', 'Short-story', 'touchingly', 'mending', 'communication', 'Eudora',
                        'Welty', 'Start', 'signing', 'dotted', 'mischievous', 'oodles', 'trumpet', 'a-bornin', 'strays',
                        'humorously', 'tendentious', 'intervention', 'who-wrote-Shakespeare', 'Neatly', 'feral',
                        'Halle', 'practiced', 'expanse', 'distaste', 'Graced', 'sheets', 'concepts', 'dreaming',
                        'checking', 'Germanic', 'quirkiness', 'energizes', 'chomps', 'tolerable', 'well-trod',
                        'barbers', 'streetwise', 'McLaughlin', 'reality-snubbing', 'Alias', 'Betty', 'visualizing',
                        'punitive', 'eardrum-dicing', 'screeching-metal', 'smashups', 'odd-couple', 'sniping', 'Reyes',
                        'wrapping', 'dearly', 'athletic', 'exploits', 'awed', 'sportsmen', 'Longley', 'tamer', 'adhere',
                        'laws', 'agile', 'Rodan', 'league', 'Truly', 'rogue', 'assassins', 'agency', 'boss', 'amnesiac',
                        'Damon\\/Bourne', 'Drawing', 'romanticism', 'sultry', 'beer-fueled', 'retiring', 'dissipated',
                        'masked', 'injustice', 'Campbell', 'outlet', 'flick-knife', 'diction', 'Swanson', 'Plot',
                        'Depicts', 'sorriest', 'jelly', 'bouncing', 'characterizes', 'recycle', 'Outside', 'unclear',
                        'undertaken', 'career-defining', 'Ace', 'Ventura', 'Pollak', 'wrestler', 'Chyna', 'Dolly',
                        'Parton', 'Imposter', 'Leery', '?!?', 'unconned', 'overmanipulative', 'practices',
                        'fourteen-year', 'Ferris', 'spirals', 'thuds', 'Cool-J', 'shootings', 'California', 'Fangoria',
                        'subscriber', 'decline', 'candor', 'dazed', 'pornography', 'Burke', 'Monster', 'horns',
                        'togetherness', 'inter-family', 'explanation', 'Splendidly', 'adversity', 'drang', 'Fuhrman',
                        'posthumously', 'published', 'checkout', 'Extraordinary', 'Josh', 'coasting', 'A.S.', 'head-on',
                        'trading', 'reverence', 'clock', 'Seeing', 'position', 'undeterminable', 'dolorous', 'trim',
                        'Freudianism', 'SOOOOO', 'low-wattage', 'badder', 'Mitchell', 'handily', 'Notting', 'heft',
                        'non-Bondish', 'jargon', 'virulently', 'Seen', 'tantamount', 'Sept.', 'Star\\/producer',
                        'infused', 'neo-realist', 'tidal', 'larky', 'belong', 'Orders', 'nurtures', 'multi-layers',
                        'orders', 'Kirshner', 'Monroe', 'out-bad-act', 'Sly', 'manager', 'Limps', 'squirm-inducing',
                        'fish-out-of-water', 'unsatisfied', 'marveled', 'flames', 'hand-drawn', 'sickly',
                        'mind-destroying', 'converts', 'Daddy', 'anarchic', 'Gilmore', 'ballast', 'Harrison', 'etched',
                        'undertaking', 'coheres', 'co-writer\\/director', 'expanded', 'J.R.R.', 'Middle-earth',
                        'Schiffer', 'Hossein', 'Amini', 'reconceptualize', 'barriers', 'porno', 'bare-bones', 'outline',
                        'Jean-Claud', 'Damme', 'Segal', 'shmear', 'goo', 'releases', 'Everyman', 'calling',
                        'painstaking', 'imparting', 'invitingly', 'overture', 'pathos-filled', 'conducted', '3-D',
                        'orbit', 'stanzas', 'Savage', 'Garden', 'resume', '99', 'Malle', 'messing', 'Slob',
                        'reductions', 'Runyon', 'crooks', 'Clumsy', 'warriors', 'ill-timed', 'Antitrust', 'fraction',
                        'Hunger', 'complicate', 'Seeks', 'quasi-Shakespearean', 'misogynist', 'breasts', 'championship',
                        'conflicted', 'spiritually', 'sixth-grade', 'height', 'drive-by', 'stud', 'knockabout',
                        'affectation', 'bravura', 'consummate', 'passe', 'chopsocky', 'brittle', 'compelled',
                        'non-Britney', 'harm', 'halfhearted', '8th', 'delving', 'asked', 'belief', 'washout', 'Putting',
                        'murderer', 'Pandora', 'Box', 'gamut', 'cheesier', 'Except', 'profits', 'efficient',
                        'eagerness', 'mistaken-identity', 'film-culture', 'referential', 'not-so-funny', 'analyze',
                        'Friggin', 'fencing', 'eminently', 'Richly', 'suggestive', 'readings', 'unfolding',
                        'junior-high', 'millisecond', 'occasion', 'overload', 'irreparably', 'useful', 'Sommers',
                        'title-bout', 'doting', 'Twenty-three', 'Gregory', 'Hinton', 'strong-minded', 'viewpoint',
                        'Dull', 'nuts', 'cold-fish', 'scrap', 'darkest', 'binging', 'void', 'Fresh', 'blown-out',
                        'walking-dead', 'cop-flick', 'brainy', 'Mrs.', 'founders', 'preciousness', 'perspicacious',
                        'blatantly', 'laugh-a-minute', 'magician', 'derailed', 'quasi-improvised', 'made-for-TV',
                        'wildcard', 'gracious', 'ray', 'forever', 'catalyst', 'restrictive', 'convinces', 'secretly',
                        'unhinged', 'faulty', 'glows', 'conniving', 'swirl', 'unspeakably', 'reams', 'remakes',
                        'Avengers', 'delinquent', 'paperbacks', 'Leather', 'Warriors', 'Switchblade', 'Sexpot',
                        'raucous', 'regrets', 'candid', 'archly', 'fruition', '2-day', 'Coke', 'so-five-minutes-ago',
                        'overwhelm', 'Hanna-Barbera', 'intermediary', 'triteness', 'device', 'Fairy-tale', 'skeleton',
                        'ugly-duckling', 'accidental', 'Painfully', 'Admirers', 'relieved', 'R', 'Xmas', 'Exploits',
                        'headbanger', 'built-in', 'drying', 'unconcerned', 'ingest', 'explicit', 'libido', 'Portugal',
                        'Tonto', 'multi-layered', 'humanist', 'affects', 'geographical', 'displacement', 'voyeuristic',
                        'dutifully', 'heartstrings', 'Deblois', 'Sanders', 'frailty', 'fascinates', 'traffics',
                        'encumbers', 'Harmless', 'agape', 'punched', 'kibosh', 'ideology', 'evoke', 'bustling',
                        'jostles', 'India', 'Gulzar', 'Jagjit', 'Singh', 'reacting', 'urgently', 'mercy', 'gasping',
                        'opulent', 'lushness', 'peevish', 'choke', 'yank', 'newness', 'Moves', 'message-mongering',
                        'moralism', 'obscured', 'fanatic', 'disgusted', 'unshapely', 'Waking', 'vowing', 'stab',
                        'hooliganism', 'double-barreled', 'shootout', 'Meat', 'Loaf', 'explodes', 'astounds', 'oomph',
                        'Sluggish', 'tonally', 'Extremities', 'dour', 'consistency', 'parsing', 'barn-side', 'breach',
                        'DiCaprio', 'Winds', 'narcissistic', 'candy-coat', 'waif', 'smear', 'lip-gloss', '1993',
                        'wazoo', 'grand-scale', 'awarded', 'subjected', 'urine', 'semen', 'substances', 'sunbaked',
                        'summery', 'ethnography', 'deceit', 'Phantom', 'Menace', 'colonialism', 'Khouri', 'continuum',
                        'elephant', 'even-handedness', 'rigged', 'emerged', 'pretence', 'Gulpilil', 'commanding',
                        'compass', 'avarice', 'nowheresville', 'depends', 'plunge', 'describing', 'invented', 'cutter',
                        'Estela', 'hagiographic', 'leader', 'Fidel', 'dwells', 'crossing-over', 'mumbo', 'Sirk',
                        'differently', 'drunk', 'sober', 'quickie', 'afterlife', 'communications', 'trope', 'judge',
                        'Weddings', 'Funeral', 'diminishing', 'stature', 'Oscar-winning', 'lowly', 'video-shot',
                        'dogged', 'kids-and-family-oriented', 'residents', 'Copenhagen', 'befuddling', 'nymphette',
                        'salt-of-the-earth', 'mommy', 'Minnie', 'Slim', 'incognito', 'wig', 'naval', 'personnel',
                        'Diego', 'welled', 'outrageously', 'ut', 'parachutes', 'unstoppable', 'superman', 'Balzac',
                        'Seamstress', 'excite', 'churns', 'flagrantly', 'thunderstorms', 'imaginary', 'frittered',
                        'chewy', 'dimming', 'recalls', 'neorealism', 'one-room', 'self-control', 'redundancy', 'Hunnam',
                        'twinkling', 'extends', 'supple', 'athlete', 'indomitability', 'Sports', 'poking', 'genitals',
                        'fruit', 'pies', 'long-running', 'half-dozen', 'palm', 'appealingly', 'bard', 'equipment',
                        'aisle', 'walker', 'superlative', 'cloaked', 'euphemism', 'believes', 'Addams', 'Boyd',
                        'Guardian', 'Avon', 'two-way', 'time-switching', 'myopic', 'stalls', 'buoy', 'surehanded',
                        'knockoff', 'betting', 'perennial', 'Everlyn', 'Sampi', 'radiates', 'star-power', 'Woodman',
                        'pipeline', 'hitch', 'zealously', 'spreading', 'Puritanical', 'Seas', 'islanders', 'relish',
                        'battles', 'Producer', 'Penotti', 'surveyed', 'Neeson', 'capably', 'fleeing', 'One-of-a-kind',
                        'frightful', 'owed', 'starving', 'Glizty', 'hard-hearted', 'Stage', 'hit-man', 'jump', 'wreck',
                        'catastrophic', 'gall', 'clamoring', 'studied', 'dependent', 'amoral', 'Bouquet', 'masterly',
                        'brazenly', 'Renaissance', 'goggles', 'hands-off', 'Annie', 'Crispin', 'Glover', 'screwing',
                        'riddle', 'enigma', 'agnostic', 'carnivore', 'pronounced', 'Pythonesque', 'item', 'superhero',
                        'squabbling', 'spouses', 'fortune', 'salaries', 'spliced', 'Midnight', '48', 'in-your-face',
                        'perpetually', 'hazy', 'civilization', 'blacklight', 'Pink', 'Floyd', 'dismissive', 'ragbag',
                        'Conceptually', 'living-room', 'Worlds', 'Disgusting', 'fabulous', 'noteworthy', 'filler',
                        'welt', 'Johnny', 'Knoxville', 'riot-control', 'projectile', 'bliss', 'nonconformity',
                        'glancing', 'Hibiscus', 'grandly', 'ministers', 'Bible-study', 'discuss', 'norm', 'Lovingly',
                        'stickiness', 'true-to-life', 'unadorned', 'rural', 'first-timer', 'Hilary', 'Birmingham',
                        'Rosenthal', 'generating', 'Helms', 'anti-', 'included', 'hue', 'drastic', 'iconography',
                        'distort', 'reassembled', 'cutting-room', 'Staggeringly', 'averse', 'follow-your-dream',
                        'giggly', 'muy', 'loco', 'Altman-esque', 'induce', 'fright', 'Breheny', 'lensing', 'Cook',
                        'drill', 'Major', 'League', 'concentrating', 'hostile', 'clause', 'Potter', 'J.K.', 'live-wire',
                        'amaze', 'rejection', 'Weirdly', 'unveil', 'wheedling', 'pointing', 'smeared', 'windshield',
                        'Blow', 'Boyz', 'Hood', 'malleable', 'ranges', 'bodacious', 'wheezy', 'P.T.', 'grandness',
                        'equalizer', 'ills', 'joys', 'under-7', 'gender-bending', 'party-hearty', 'scalds', 'acid',
                        'bowling', 'McDowell', 'dislikable', 'sociopathy', 'cadavers', 'Zoe', 'Clarke-Williams',
                        'enemies', 'transfigures', 'unchanged', 'Exploitative', 'bearable', 'operative', 'imitations',
                        'subsided', 'companionable', 'titans', 'proclaim', 'love-struck', 'somebodies', 'quashed',
                        'obscenity', 'voices-from-the-other-side', 'Off', 'Hook', 'writer-producer-director',
                        'Watstein', 'finishing', 'deny', 'sufficiently', 'unreligious', 'shear', 'drama\\/character',
                        'Kidd', 'populating', 'meanspirited', 'cash', 'Darkly', 'percentages', 'Stinks', 'burlap',
                        'gloom', 'be-bop', 'nighttime', 'videologue', 'protective', 'cocoon', 'Petter', 'NÃ¦ss', 'Axel',
                        'Hellstenius', 'doubles', 'Ghandi', 'Brothers-style', 'down-and-dirty', 'laugher', 'Blimp',
                        'ready-made', 'founding', 'Yong', 'Kang', 'Kozmo', 'dictator-madman', 'undeserved', 'springing',
                        'scummy', 'ripoff', 'Cronenberg', 'Videodrome', 'secular', 'far-fetched', 'consumerist',
                        'studiously', 'trusted', 'denouement', 'soft-core', 'Shoe', 'Diaries', 'paranormal', 'nosedive',
                        'attractions', 'Conrad', 'nominated', 'dig', 'Dungeons', 'Dragons', 'military', 'captivated',
                        'u-boat', 'severely', 'tested', 'immaturity', 'P.C.', 'portraits', 'freshening',
                        'Iranian-American', '1979', 'moldering', 'occupational', 'subsequent', 'reinvention', 'Gussied',
                        'distracting', 'shriek', 'Gay', 'dwindles', 'Whole', 'Press', 'delete', 'suspects', 'endorses',
                        'debts', 'Aliens', 'dragon', 'walks', 'louts', 'climbing', 'steps', 'stadium-seat', 'unseen',
                        'resides', 'Tornatore', 'Mom', 'Dad', '1950', 'Doris', 'Overly', 'worshipful', 'bio-doc',
                        'down-home', 'Bob', 'adept', 'Giggling', 'inconsistencies', 'hippopotamus', 'ballerina',
                        'self-exploitation', 'merge', 'soft-porn', 'powerment', 'Believer', 'wire', 'stew',
                        'critically', 'Humorous', 'kid-pleasing', 'tolerable-to-adults', 'disapproval', 'Justine',
                        'tinge', 'dry-eyed', 'moisture', 'wafer-thin', 'wades', 'Drunken', 'epiphanies', 'Needs',
                        'impressionistic', 'point-of-view', 'slow-motion', 'quick-cut', 'Patient', 'Unbearable',
                        'Lightness', 'reputedly', 'unfilmable', 'bucked', 'daredevils', 'straight-shooting', 'prepared',
                        'cling', 'engulfed', 'Shapiro', 'Goldman', 'Bolado', 'presume', 'trot', 'skimpy', 'drek',
                        'elegiac', 'slow-moving', 'police-procedural', 'Anemic', 'Partly', 'volcano', 'overflowing',
                        'septic', 'tank', 'Cuba', 'Ana', 'Swank', 'wide-awake', 'Abbass', 'elemental', 'symbols',
                        'life-at-arm', 'irritatingly', 'S&M', 'Gone', 'boho', 'Sensation', 'rebellion', 'continually',
                        'accommodate', 'make-believe', 'arctic', 'tundra', 'Transforms', 'Hence', 'Deepa',
                        'resurrecting', 'flavorful', 'Due', 'stodgy', 'TelePrompTer', 'Oscar-size', 'awakens', 'glued',
                        'transvestite', 'rake', 'relayed', 'Sharks', 'Player', 'skewering', 'insiders', 'pop-induced',
                        'eh', 'Reveals', 'straight-faced', 'preoccupations', 'Thought-provoking', 'hermetic',
                        'Worthless', 'pseudo-rock-video', 'flavours', 'computer-animated', 'serpent', '10th',
                        'dark-as-pitch', 'therapeutic', 'zap', '1933', 'beast-within', 'hack-and-slash', 'credulity',
                        'alternating', 'sloppiness', 'ooze', 'K', '19', 'drama\\/action', 'spotty', 'pearls',
                        'clicking', 'Truckzilla', 'cryin', 'rug', 'plunging', 'bloodstream', 'Loved', 'Ones',
                        'commands', 'acumen', 'warped', 'Wimmer', 'Musketeer', 'entertains', 'trusts', 'unendurable',
                        'ultra-provincial', '26-year-old', 'Reese', 'Melanie', 'Carmichael', 'End', 'Days', 'facial',
                        'lacked', 'economics', 'fabulousness', 'Xiaoshuai', 'well-realized', 'embroils', 'circular',
                        'voting', 'speedy', 'swan', 'dive', 'landing', 'twinkle', 'jungle', 'needing', 'Glazed', 'scum',
                        'Dash', 'aptitude', 'Abdul', 'Malik', 'Abbott', 'Ernest', 'Tron', 'oversized', 'bedtime',
                        'patriotic', 'strategic', 'dramatizing', 'repeating', 'well-done', 'self-reflexive',
                        'Holofcenter', 'wrap', 'tie', 'Heidi', 'Mai', 'Thi', 'defend', 'frothing', 'ex-girlfriend',
                        'Worthy', 'joint', 'promotion', 'Basketball', 'Association', 'teenaged', 'poster-boy',
                        'Hollywood-itis', 'harshness', 'shockwaves', 'gratify', 'corrupt', 'weasels', 'scenarios',
                        'sermonize', 'okay', 'restroom', 'MY', 'LITTLE', 'EYE', 'splashy', 'operational',
                        'Unsurprisingly', 'caretakers', 'Lillard', 'Shaggy', 'depressingly', 'retrograde', 'Smarter',
                        'commercials', 'disease-of-the-week', 'small-screen', 'Latino', 'ANTWONE', 'FISHER', 'Sydow',
                        'Intacto', 'Shepard', 'examples', 'Poke-mania', 'preceded', 'uncommitted', 'gullible',
                        'co-operative', 'electoral', 'broadcast', 'dupe', 'Important', 'stripe', 'whiff', 'vaunted',
                        'barn-burningly', 'speed', 'deathly', '72-year-old', 'slowed', 'fleet-footed', 'Um', 'leaps',
                        'stylishly', 'community-college', 'advertisement', 'guitar', 'amp', 'syrup', 'underwater',
                        'grizzled', 'charred', 'northwest', 'Bermuda', 'Triangle', 'amusedly', 'impatiently', 'fancies',
                        'Selby', 'ounce', 'kids-cute', 'faked', 'lucks', 'anteing', 'Playing', 'Bergmanesque',
                        'moral-condundrum', 'Schaefer', 'Transcends', 'derring-do', 'confuse', 'captive', 'grossly',
                        'excessively', 'underconfident', 'Walken', 'romanced', 'Cyndi', 'Lauper', 'Opportunists',
                        'Kjell', 'Bjarne', 'naÃ¯f', 'Piercingly', 'pander', 'basest', 'desires', 'payback',
                        'none-too-original', 'liar', 'headaches', 'treatise', 'and-miss', 'Weighty', 'anonymity',
                        'environments', 'merrily', 'turntablists', 'jugglers', 'schoolers', 'innovators', 'documenting',
                        'bytes', 'anciently', 'mÃ©tier', 'essay', 'specter', 'suicide', 'Rorschach', 'euphoria',
                        'variable', 'light-footed', 'enchantment', 'curve', 'mar', 'AAA', 'rated', 'EEE', 'startled',
                        'dozing', 'verisimilitude', 'phenomenon', 'peaked', 'bewildered', 'unpleasantly', 'in-joke',
                        'tiniest', 'coma-like', 'Doug', 'colleagues', 'boom-bam', 'eye-catching', 'mid-to-low',
                        'betrayed', 'refined', 'sloughs', 'mire', 'alleged', 'Marinated', 'Spiderman', 'ROCKS',
                        'post-September', 'reprehensible', 'manipulating', 'bestowing', 'unequivocally', 'trifecta',
                        'Janey', 'obligations', 'guessable', 'morbidity', 'jams', 'prefabricated', 'botches',
                        'principal', 'villainess', 'leanest', 'meanest', 'hungry', 'twisty', 'sewing', 'head-trip',
                        'Hailed', 'neo-Hitchcockianism', 'Chabrolian', 'Tyson', 'E', 'indulge', 'vacant', 'slave',
                        'Copmovieland', 'routes', 'laughingly', 'artwork', 'animaton', 'snobbery', 'middle-America',
                        'diversions', 'contain', 'Balto', 'Toy', 'futility', 'commander-in-chief', 'propels',
                        'echelons', 'plumbing', 'existing', 'cooly', 'review', 'backbone', 'boom', 'mikes',
                        'flag-waving', 'cautions', 'lushly', 'recorded', 'Argentinian', 'Predecessors', 'masterpieces',
                        'duel', 'controlled', 'pan-American', 'cruelly', 'burning', 'blasting', 'stabbing',
                        'self-critical', 'behind-the-scenes', 'navel-gazing', 'Orleans', 'infinite', 'Nonchalantly',
                        'Check', 'decoder', 'Shakes', 'Clown', 'self-righteous', 'trove', 'heels', 'necessity',
                        'Towers', 'outdoes', 'graphics', 'Talky', 'Kafkaesque', 'overnight', 'robbed', 'persecuted',
                        'Sinks', 'court', 'maneuvers', 'Alternates', 'introspection', 'Shum', 'kilt-wearing',
                        'theatrically', 'Barbarian', 'revisiting', 'improvise', 'directionless', 'Violent',
                        'forgettably', 'Harmon', 'less-is-more', 'bump-in', 'the-night', 'crawlies', 'helpful',
                        'extremist', 'name-calling', 'calculating', 'fiend', 'self-promoter', '98', 'bratty', 'Close',
                        'freewheeling', 'trash-cinema', 'every-joke-has', 'been-told-a', 'thousand-times', 'Pie-like',
                        'irreverence', 'Purposefully', 'eroticized', 'ultra-violent', 'inferior', 'Solomonic',
                        'tearing', 'cue', 'Sterile', 'Color', 'Or', 'Warmth', 'evergreen', 'exuberant', 'expresses',
                        'late-night', 'sleepwalk', 'vulgarities', 'refuse', 'arguing', 'ABC', 'whopping', 'shootouts',
                        'fine-looking', 'clutch', 'punctuation', 'salvos', 'idiotically', 'feardotcom.com',
                        'improperly', 'Rea', 'Applegate', 'Posey', 'you-are-there', 'uncouth', 'vicious', 'Claims',
                        'simpering', 'Lasker', 'distances', 'Byron', 'Luther', 'satisfaction', 'Theater',
                        'well-defined', 'torments', 'operatic', 'mantra', 'Thurman', 'career-best',
                        'hyper-artificiality', 'racism', 'homophobia', 'spied', 'cookie-cutter', 'confluence',
                        'surface-effect', '19th', 'Always', 'Asquith', 'acclaimed', 'flounders', 'Brave', 're',
                        'enactments', 'Degenerates', 'hogwash', 'bless', 'aversion', 'cashing', 'gorgeousness',
                        'venturesome', 'utilizing', 'moist', 'unrepentant', 'epicenter', 'percolating', 'instability',
                        'overripe', 'mouthpieces', 'motifs', 'pretention', 'jaunt', 'lane', 'televised', 'Bart',
                        'anchoring', 'melancholia', 'gut', 'shaky', '1790', 'imagines', 'naturalness', 'superheroics',
                        'engross', 'antsy', 'Speaks', 'symbiotic', 'grabs', 'shakes', 'vigorously', 'mount', 'cogent',
                        'defense', 'marveling', 'superhuman', 'withstand', 'chefs', 'fussing', 'anarchy', 'strangling',
                        'eight', 'boarders', 'co-winner', 'Audience', 'completion', 'artworks', 'Distinctly', 'sub-par',
                        'shivers', 'crappy', 'newcomers', 'ineffective', 'Strip', 'debris', 'four-star', 'seizing',
                        'haplessness', 'tics', 'Bertrand', 'oft-brilliant', 'Combine', 'claustrophobia', 'sooner',
                        'straight-to-video', 'bigger-name', 'Silly', 'nurses', 'gaping', 'Olympic', 'brain-deadening',
                        'hangover', 'participant', 'government', 'Marine\\/legal', 'subliminally', 'province',
                        'Shadows', 'Motown', 'depleted', 'brand-new', 'smitten', 'troubadour', 'acolytes',
                        'damaged-goods', 'orbits', 'mortal', 'awareness', 'spooks', 'rancorous', 'Sensitive',
                        'outlandish', 'small-scale', 'Verdu', 'mourns', 'cesspool', 'lessen', 'Brisk', 'build-up',
                        'expository', 'Staggers', 'madcap', 'thinly-conceived', 'acquainted', 'astronomically',
                        'misconceived', 'OS', 'introduces', 'thinkers', 'ethnicities', 'swiftly', 'maintained',
                        'excellence', 'Execrable', 'mind-bender', 'pabulum', '95-minute', 'NBA', 'properties', 'Fraser',
                        'righteousness', 'jostling', 'elbowed', 'rotoscope', 'cheap-looking', 'Candid', 'Camera',
                        'lower-wit', 'oeuvre', 'defines', 'fizzle', 'Norma', 'Rae', 'Pierce', 'Romoli', 'super-powers',
                        'super-simple', 'super-dooper-adorability', 'war-movie', 'exquisitely', 'culture-clash',
                        'discomfort', 'Europe', 'self-serious', 'savagely', 'Swift', 'philosophers', 'flop', 'steam',
                        'miserably', 'dreadfulness', 'strips', 'sanctimoniousness', 'church-wary', '24-and-unders',
                        'Caddyshack', 'adopt', 'generational', 'Snoots', 'standbys', 'malarkey', 'gobbler', 'claims',
                        'ashamed', 'admitting', 'Downbeat', 'period-perfect', 'hammers', 'Hard-core', 'confusion',
                        'bailiwick', 'unseemly', 'congratulate', 'goofiest', 'couch', 'Dr.', 'beloved-major',
                        'character-who-shall', 'remain-nameless', 'invite', 'laser', '78', 'vibrance', 'Ourside',
                        'spending', 'Kalvert', 'gangs', '1958', '15th', 'Arkansas', 'consists', 'truck-loving', "ol'",
                        'peroxide', 'blond', 'honeys', 'worldly', 'supermarket', 'tabloids', 'fell', 'masterpeice',
                        'Fulford-Wierzbicki', 'wise-beyond-her-years', 'single-minded', 'step-printing', 'inadequate',
                        'Begins', 'docu-drama', 'multi-character', 'flourish', 'counting', 'bolt', 'Wonton', 'floats',
                        'capitalize', 'Lecter', 'asylum', 'Jia', 'guffaw', 'triumphant', 'frontman', 'Choppy',
                        'Lifestyle', 'perfervid', 'ces', 'imperious', 'Katzenberg', 'Prince', 'Egypt', '1998',
                        'not-at-all-good', 'drifts', 'cracks', 'ever-growing', 'unembarrassing', 'juiced', 'all-French',
                        'marveilleux', 'Patchy', 'low-tech', 'academic', 'skullduggery', 'word-of-mouth',
                        'underdeveloped', 'conveyor', 'Character', 'drowsy', 'infatuated', 'self-examination',
                        'kiddie-oriented', 'gum', 'sneak', 'encyclopedia', 'shoplifts', 'farewell-to-innocence',
                        'Wanderers', 'Bronx', 'Tale', 'cribbing', 'civil', 'disobedience', 'anti-war', 'star-making',
                        'machinery', 'tinseltown', 'Gaping', 'Mazel', 'tov', 'subzero', 'overboard', 'rapport',
                        'HÃ©lÃ¨ne', 'Cranky', 'camps', 'fringe', 'theorist', 'unlikeable', 'violated', 'targets',
                        'raunch-fests', 'unmentionables', 'cleverest', 'veins', 'Granger', 'Gauge', 'punny', '6',
                        'grade-school', 'retro-refitting', 'recall', 'Nickelodeon-esque', 'rooted', 'undergoing',
                        'label', 'Milder', 'jangle', 'contender', 'standoffish', 'hereby', 'Passionate', 'tarantula',
                        'Helga', 'prominently', 'rugrats', 'Ghostbusters', 'architecture', 'pros', 'cons', 'duties',
                        'gutless', 'Laurice', 'Guillen', 'spreads', 'Margolo', 'insensitivity', 'rarest', 'kinds',
                        'family-oriented', 'non-Disney', 'Liana', 'Dognini', 'Irvine', 'Trainspotting', 'reporters',
                        'willingly', 'posterity', 'big-budget\\/all-star', 'unblinkingly', 'ballsy', 'competing',
                        'lawyers', 'stomach-knotting', 'undertone', 'Brent', 'Hanley', 'huskies', 'border', 'collie',
                        'lingual', 'wiggling', 'kitten', 'frosting', 'bowl', 'hotels', 'highways', 'moan', 'entrapment',
                        'gymnastics', 'oily', 'dealer', 'pile-ups', 'engorged', 'react', 'Anton', 'ick', 'depicted',
                        'managing', 'Jake', 'rise-and-fall', 'glamour', 'Contradicts', 'nowadays', 'Absorbing',
                        'AndrÃ©', 'daft', 'lays', 'interweaves', 'Mobius', 'elliptically', 'loops', 'began',
                        'computerized', 'Yoda', 'ascends', 'cranky', 'rediscover', 'quivering', 'civilized',
                        'amusement', 'funk', 'tearful', 'pessimism', 'contradicts', 'aspired', 'ivans', 'xtc', 'notch',
                        'renegade-cop', 'Thought', 'Awful', 'duking', 'indieflick', 'trims', 'Mention', 'underrated',
                        'receive', 'sunset', 'achival', 'Metro', 'strategies', 'convolutions', 'plausible',
                        'zinger-filled', 'open-minded', 'Infidelity', 'well-edited', 'standout', 'partisans',
                        'sabotage', 'Gaunt', 'silver-haired', 'Sandeman', 'negatives', 'outweigh', 'positives',
                        'olives', '1962', 'garnish', 'Meandering', 'teenager', 'self-inflicted', 'subtitled',
                        'internal', 'combustion', 'engine', 'thump', 'UB', 'spoofs', 'outre', 'dorkier', 'inseparable',
                        'cooks', 'smoky', 'conceptions', 'Brutally', 'Aggressive', 'whitewash', 'jumbled',
                        'unapologetic', 'sweetheart', 'Mandy', 'derives', 'slights', 'exchanges', 'Beating',
                        'downplays', 'outrageousness', 'staying', 'circuit', 'wind-in-the-hair', 'Rocawear', 'provoked',
                        'Unwieldy', 'contraption', 'restatement', 'validated', 'Spade', 'Citizen', 'harangues',
                        'Ontiveros', 'Chances', 'remembrance', 'reunions', 'gender-provoking', 'submerged', 'admirers',
                        'degrading', 'dispassionate', 'Gantz', 'pasty', 'lumpen', 'coolness', 'tunnels', 'attackers',
                        'surveillance', 'technologies', 'Roach', 'taxicab', 'critiquing', 'faltering', 'half-step',
                        'criticizing', 'commiserating', 'Elfriede', 'Jelinek', 'unmotivated', 'unpersuasive',
                        'inconclusive', 'write', 'liner', 'spiteful', 'salvaged', 'Occasionally', 'waking', 'equate',
                        'twenty', 'east', 'welcomed', 'tens', 'democracie', 'influential', 'minus', 'made-up',
                        'refreshes', 'Usual', 'Suspects', 'march', 'drum', 'housing', 'options', 'e-graveyard', 'Chin',
                        'skid-row', 'margins', 'Ong', 'well-told', 'overburdened', 'second-guess', 'dudsville',
                        'enforcement', 'Hades', 'Excruciatingly', 'unromantic', '2455', 'crossed', 'messenger',
                        'Borscht', 'Belt', 'schtick', 'inherently', 'justifying', 'spikes', 'self-absorbed', 'featured',
                        'descriptions', 'besotted', 'inventing', 'theorizing', 'German-Expressionist', 'according',
                        'render', 'loop', 'reopens', 'flower-power', 'compendium', 'two-hour-and-fifteen-minute',
                        'Toro', 'closed-door', 'hanky-panky', 'juiceless', 'uninventive', 'aspire', 'lethargic',
                        'peppered', 'Mama', 'Africa', 'feel-bad', 'items', 'winking', 'heap',
                        'coming-of-age\\/coming-out', 'classy', 'Notwithstanding', 'SECRETARY', 'Spader', 'vacuous',
                        'embody', 'nouvelle', 'stiflingly', 'ripper', 'chill', 'ace', 'Japanimator', 'Hayao',
                        'survives', 'BV', 're-voiced', 'Throwing', 'pin', 'grenade', 'ransacked', 'baked', 'assures',
                        'pocket', 'keel', 'Play-Doh', 'overstated', 'Alternative', 'picaresque', 'little-remembered',
                        'avert', 'boorish', 'I-heard-a-joke', 'at-a-frat-party', 'Aldrich', 'big-hearted', 'kiddies',
                        'masala', 'Replacing', 'tracking', 'handheld', 'video-cam', 'acquire', 'conversation',
                        'humbling', 'fueled', 'Zhao', 'Benshan', 'Jie', 'complain', 'bedeviled', 'Fiji', 'diver',
                        'Rusi', 'Vulakoro', 'Michelle', 'Resembles', 'Somewhere', 'Rosemary', 'Baby', 'well-conceived',
                        'Informative', 'discussed', 'Dismally', 'Watchable', 'Initially', 'POW', 'irreparable',
                        'analysis', 'offset', 'intellectuals', 'nonbelievers', 'rethink', 'attitudes', 'creed',
                        'skeptics', 'Engages', 'unfairly', 'fabricated', 'howler', 'woods', 'pseudo-educational',
                        'Formuliac', 'debilitating', 'aftermath', 'attacks', 'First-timer', 'Idiotic', 'superficially',
                        'ramble', 'remarks', 'engendering', 'probation', 'officer', 'unsuccessful', 'Sillier', 'cuter',
                        'EXIT', 'Shrewd', 'invites', 'unflattering', 'comparisons', 'installments', 'registering',
                        'Loses', 'twitchy', 'boorishness', 'Juliet', 'Stevenon', 'Pamela', 'roller', 'coaster', 'tides',
                        'single-handedly', 'journalists', 'campaign', 'publicity', 'talked', 'Christ', 'foundation',
                        'setpiece', 'unified', 'ex-Marine', 'Generates', 'pixilated', 'dispel', 'tooled', 'volumes',
                        'dreamscape', 'frustrates', 'captivates', 'Matters', '1937', 'Dwarfs', 'Leys', 'climb',
                        'substitutes', 'ruthless', 'hidebound', 'movie-making', 'WEIRD', 'stitch', 'beware', 'penance',
                        'Insufferably', 'inflammatory', 'hatred', 'Elaborate', 'centre', 'in-jokey', 'Boomers',
                        'Barrie', 'highly-praised', 'misfortune', 'broaches', 'neo-Augustinian', 'theology',
                        'best-known', 'ham', 'Opening', 'closed', 'fantasti', 'orientation', 'touchstone', 'Asks',
                        'non-firsthand', 'liveliness', 'ear-pleasing', 'Savvy', 'day-old', 'location', 'Truth',
                        'Consequences', 'N.M.', 'interchangeable', 'actioner', 'imbecilic', 'Mafia', 'toolbags',
                        'botching', 'assignment', 'backwater', 'Chicago-based', 'old-world', 'election', 'graphically',
                        'fledgling', 'democracies', 'ballot', 'Bladerunner', 'sci', 'fi', 'whistle',
                        'Co-writer\\/director', 'Brazil-like', 'hyper-real', 'Plutonium', 'Circus', 'Purgatory',
                        'contentious', 'configurations', 'Criminal', 'apartheid', 'Upsetting', 'singer-turned', 'no.',
                        'mateys', 'gawky', 'Spall', 'Heathers', 'divided', 'numbness', 'resumes', 'cleaving',
                        'oversimplification', 'Pax', 'lulled', 'Written', 'Kendall', 'Decter', 'Stortelling',
                        'oftentimes', 'cowardly', 'autocritique', 'rhapsodic', 'Werner', 'shapes', 'Effective',
                        'auspicious', 'middling', 'Clever', 'barbs', 'southern', 'Adrian', 'sleazy', 'Thoughtless',
                        'Scouse', 'far-flung', 'illogical', 'prescribed', 'soothing', 'Muzak', 'cushion', 'Hate',
                        'Holland', 'peter', 'burgeoning', 'sent', 'alterations', 'motorized', 'scooter', 'dewy-eyed',
                        'kid-movie', 'G-rated', 'squirming', 'Feral', 'half-an-hour', 'furious', 'ninety',
                        'often-hilarious', 'bang-the-drum', 'emphatic', 'Translating', 'miracles', 'beyond-lame',
                        'Teddy', 'Picnic', 'debuts', 'esteemed', 'writer-actor', 'dismantle', 'facades', 'wonderous',
                        'accomplishment', 'veracity', 'vaporize', 'Twinkie', 'pageantry', 'roars', 'eroticism',
                        'pardon', 'goth', 'sillified', 'stop-and-start', 'non-mystery', 'bombards', 'BMW', 'haranguing',
                        'Sparse', 'Adults', 'Bergman', 'fatalism', 'Larson', 'forward', 'Sally', 'Raphael',
                        'Philadelphia', 'ailments', 'uninflected', 'fax', 'anthropomorphic', 'Italian-language',
                        'Officially', 'bestial', 'Slow', 'indoor', 'thesps', 'slumming', 'enthusiastically', 'invokes',
                        'percussion', 'brass', 'football']

        final_words = []
        for word in temp_sentence:
            if not word.istitle():
                if len(word) < 17:
                    word = word.lower()
                    if word not in stopwords and word not in punctuations:
                        final_words.append(word)
        index_of_words = [self.indexer.add_and_get_index(word, add=add_to_indexer) for word in final_words]

        # Initiating the feature size to be 14,000 words
        feature = np.zeros(self.size())
        for i in index_of_words:
            if i < self.size():
                feature[i] += 1
        return feature


"""
        final_words = []
        for word in temp_sentence:
            word = word.lower()
            if word not in stopwords:
                if word not in low_frequency_words:
                    final_words.append(word)
                else:
                    low_frequency_words.remove(word)
        index_of_words = [self.indexer.add_and_get_index(word, add=add_to_indexer) for word in final_words]

        #Initiating the feature size to be 14,000 words
        feature = np.zeros(self.size())
        for i in index_of_words:
            if i <self.size():
                feature[i] +=1
        return feature
"""


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """

        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        weight_vector = np.zeros(self.feat_extractor.size())
        epoch = 12 if self.feat_extractor.size() == 15000 else 9
        for i in range(epoch):
            random.seed(3)
            random.shuffle(train_exs)
            number_training_examples = len(train_exs)
            matched = np.zeros(number_training_examples)
            for k in range(number_training_examples):
                feature_vector = self.feat_extractor.extract_features(train_exs[k], True)
                pred = float(np.dot(weight_vector, feature_vector)>0)
                if pred == train_exs[k].label:
                    matched[k] = 1
                    continue
                else:
                    if pred == 0 and train_exs[k].label == 1:
                        weight_vector = weight_vector + 1.25*feature_vector
                    else:
                        weight_vector = weight_vector - 1.25*feature_vector
            print('epoch count: %s, matched percentage: %.6f' % (i, np.mean(matched)))
        self.weight_vector = weight_vector

    def predict(self, sentence: List[str]) -> int:
        feature = self.feat_extractor.extract_features(sentence)
        if np.dot(self.weight_vector, feature) > 0:
            return 1
        else:
            return 0

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, train_exs, feat_extractor):
        self.feat_extractor = feat_extractor
        weight_vector = np.zeros(self.feat_extractor.size())
        epoch = 13 if self.feat_extractor.size() == 15000 else 9
        for i in range(epoch):
            random.seed(1)
            random.shuffle(train_exs)
            number_training_examples = len(train_exs)
            matched = np.zeros(number_training_examples)
            for k in range(number_training_examples):
                feature_vector = self.feat_extractor.extract_features(train_exs[k], True)
                pred_variable = np.dot(weight_vector, feature_vector)
                pred = float(np.exp(pred_variable)/(1+np.exp(pred_variable)) > 0.5)
                Prob_Y_1_given_x = np.exp(pred_variable)/(1+np.exp(pred_variable))
                if pred == train_exs[k].label:
                    matched[k] = 1
                    continue
                else:
                    if pred == 0.0 and train_exs[k].label == 1:
                        weight_vector = weight_vector + 0.0005*feature_vector*(1-Prob_Y_1_given_x)
                    else:
                        weight_vector = weight_vector - 0.0005*feature_vector*(Prob_Y_1_given_x)
            print('epoch count: %s, matched percentage: %.6f' % (i, np.mean(matched)))
        self.weight_vector = weight_vector

    def predict(self, sentence: List[str]) -> int:
        feature = self.feat_extractor.extract_features(sentence)
        pred_variable = np.dot(self.weight_vector, feature)
        Prob_Y_1_given_x = np.exp(pred_variable) / (1 + np.exp(pred_variable))
        if Prob_Y_1_given_x > 0.5:
            return 1
        else:
            return 0


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    perceptron_model = PerceptronClassifier(train_exs, feat_extractor)
    return perceptron_model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    LR_model = LogisticRegressionClassifier(train_exs, feat_extractor)
    return LR_model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model

