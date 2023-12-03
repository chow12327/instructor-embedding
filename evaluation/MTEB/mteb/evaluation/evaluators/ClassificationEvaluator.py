import logging

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor

from .Evaluator import Evaluator

import checklist
from checklist.editor import Editor
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.test_suite import TestSuite
from checklist.expect import Expect
import spacy
import pandas as pd

logger = logging.getLogger(__name__)

DEFINITIONS = {
    'hkunlp/instructor-base':{
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying if the sentence is positive or not: ',
        'TweetSentimentExtractionClassification': 'Represent a Twitter sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent an amazon sentence for classifying whether the sentence is counterfactual or not: ',
        'ImdbClassification': 'Represent an amazon review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'represent the sentence for classifying its purpose asweather_query, '
                                         'audio_volume_other, iot_cleaning, '
                                         'datetime_query, lists_remove, lists_createoradd, '
                                         'datetime_convert, play_music, iot_hue_lightdim, '
                                         'calendar_remove, iot_coffee, '
                                         'general_greet, alarm_query, calendar_set, '
                                         'recommendation_locations, '
                                         'lists_query or email_query: ',
        'MassiveScenarioClassification':  "Represent a scene for determining the scene as"
                                         "play, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "email, cooking, news or question answering: ",
        'MTOPDomainClassification': 'represent a sentence; ',
        'MTOPIntentClassification': 'Represent the sentence for determining its purpose as question_music, '
                                    'start_shuffle_music, get_call_time, '
                                    'get_reminder_location, is_true_recipes, '
                                    'ignore_call, get_contact_method, '
                                    'update_reminder, delete_alarm, set_default_provider_music, '
                                    'end_call, '
                                    'skip_track_music, create_timer, cancel_message, '
                                    'get_category_event, repeat_all_off_music, get_timer, '
                                    'add_time_timer, resume_music, add_to_playlist_music, update_reminder_location, '
                                    'set_rsvp_interested, pause_timer, update_timer, play_media, replay_music: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification': 'represent an amazon review sentence: ',
    },
    'hkunlp/instructor-large':{
        'Banking77Classification': 'Represent the bank purpose for classification: ',
        'EmotionClassification':  'Represent an emotion sentence for classifying the emotion: ',
        'TweetSentimentExtractionClassification': 'Represent a Tweet sentence for classification: ',
        'AmazonCounterfactualClassification': 'Represent a counter-factual sentence for classification: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for classifying the purpose as one of qa_maths, takeaway_order, weather_query, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                       'iot_hue_lighton, iot_hue_lightchange, alarm_remove, play_radio, '
                                       'transport_taxi, datetime_query, lists_remove, lists_createoradd, '
                                       'datetime_convert, play_music, iot_hue_lightdim, email_querycontact, qa_factoid, '
                                       'cooking_query, music_query, qa_currency, calendar_query, music_settings, '
                                       'music_dislikeness, audio_volume_mute, cooking_recipe, general_joke, play_game, '
                                       'news_query, recommendation_events, music_likeness, audio_volume_down, '
                                       'calendar_remove, iot_coffee, transport_traffic, iot_wemo_off, email_sendemail, '
                                       'iot_hue_lightup, social_query, social_post, iot_hue_lightoff, transport_query, '
                                       'general_greet, play_podcasts, alarm_query, calendar_set, alarm_set, '
                                       'transport_ticket, general_quirky, audio_volume_up, iot_wemo_on, qa_definition, '
                                       'recommendation_locations, play_audiobook, email_addcontact, takeaway_query, '
                                       'lists_query or email_query: ',
        'MassiveScenarioClassification': "Represent the scene for classifying its scene as one of calendar, "
                                         "play, general, alarm, music, iot, audio, takeaway, datetime, recommendation, "
                                         "social, lists, email, transport, cooking, weather, news or qa: ",
        'MTOPDomainClassification': 'Represent a sentence: ',
        'MTOPIntentClassification': 'Represent the sentence for retrieval: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        'AmazonReviewsClassification 42.12': 'Represent a review sentence for classification: ',
        'AmazonReviewsClassification': 'Represent a review for classification: ',
    },
    'hkunlp/instructor-xl': {
        'Banking77Classification': 'Represent the bank77 purposes for retrieving its bank intent: ',
        'EmotionClassification':  'Represent the amazon emotion sentence for classifying the emotion: ',
        'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        # 'AmazonCounterfactualClassification': 'Represent Daily casual counter-sentences for categorization as correct-sentences or counter-sentences: ',
        'ImdbClassification': 'Represent a review sentence for classifying emotion as positive or negative: ',
        'MassiveIntentClassification':'Represent the sentence for categorizing its task intent as qa_maths, takeaway_order, '
                                       'audio_volume_other, recommendation_movies, iot_cleaning, qa_stock, '
                                      'or recommendation_locations: ',
        'MassiveScenarioClassification': "represent an ms sentence for retrieving its intent: ",
        'MTOPDomainClassification': 'represent a MTO sentence to retrieve the task intention: ',
        'MTOPIntentClassification': 'Represent an mto sentence for retrieving its behind task intention: ',
        'ToxicConversationsClassification': 'Represent a toxicity comment for classifying its toxicity as toxic or non-toxic: ',
        # 'ToxicConversationsClassification': 'Represent toxicity comments for classifying its toxicity as toxic or non-toxic: ',
        'AmazonPolarityClassification': 'Represent the sentiment comment for retrieving a duplicate sentence: ',
        # 'AmazonPolarityClassification': 'Represent the sentiment comment to retrieve a duplicate sentence: ',
        'AmazonReviewsClassification': 'Represent an amazon review sentence to find the emoation; ',
        # 'AmazonReviewsClassification': 'Represent an amazon movie review sentence to categorize the emotion; ',
        'TweetSentimentExtractionClassification': 'Represent Daily-life spoken sentences for categorization; Input: ',
        # 'TweetSentimentExtractionClassification': 'Represent Daily-life spoken expression for classification; Input: ',
    },
}

class kNNClassificationEvaluator(Evaluator):
    def __init__(
        self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        print('use kNNClassificationEvaluator')
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean"]:  # TODO: "dot"
            knn = KNeighborsClassifier(n_neighbors=self.k, n_jobs=-1, metric=metric)
            knn.fit(X_train, self.y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            ap = average_precision_score(self.y_test, y_pred)
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            scores["ap_" + metric] = ap
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            max_ap = max(max_ap, ap)
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        scores["ap"] = max_ap
        return scores, test_cache


class kNNClassificationEvaluatorPytorch(Evaluator):
    def __init__(
        self, sentences_train, y_train, sentences_test, y_test, k=1, batch_size=32, limit=None, **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]

        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test

        self.batch_size = batch_size

        self.k = k

    def __call__(self, model, test_cache=None):
        print('use kNNClassificationEvaluatorPytorch')
        scores = {}
        max_accuracy = 0
        max_f1 = 0
        max_ap = 0
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        if test_cache is None:
            X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
            test_cache = X_test
        else:
            X_test = test_cache
        for metric in ["cosine", "euclidean", "dot"]:
            if metric == "cosine":
                distances = 1 - self._cos_sim(X_test, X_train)
            elif metric == "euclidean":
                distances = self._euclidean_dist(X_test, X_train)
            elif metric == "dot":
                distances = -self._dot_score(X_test, X_train)
            neigh_indices = torch.topk(distances, k=self.k, dim=1, largest=False).indices
            y_train = torch.tensor(self.y_train)
            y_pred = torch.mode(y_train[neigh_indices], dim=1).values  # TODO: case where there is no majority
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="macro")
            ap = average_precision_score(self.y_test, y_pred)
            scores["accuracy_" + metric] = accuracy
            scores["f1_" + metric] = f1
            scores["ap_" + metric] = ap
            max_accuracy = max(max_accuracy, accuracy)
            max_f1 = max(max_f1, f1)
            max_ap = max(max_ap, ap)
          
        scores["accuracy"] = max_accuracy
        scores["f1"] = max_f1
        scores["ap"] = max_ap
        return scores, test_cache

    @staticmethod
    def _cos_sim(a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    @staticmethod
    def _euclidean_dist(a: Tensor, b: Tensor):
        """
        Computes the euclidean distance euclidean_dist(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = euclidean_dist(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.cdist(a, b, p=2)

    @staticmethod
    def _dot_score(a: Tensor, b: Tensor):
        """
        Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        return torch.mm(a, b.transpose(0, 1))


class logRegClassificationEvaluator(Evaluator):
    def __init__(
        self,
        sentences_train,
        y_train,
        sentences_test,
        y_test,
        max_iter=100,
        batch_size=32,
        limit=None,
        checkRobustness=False,
        checkMultiLinguality=False,
        robustnessSamples=5,
        **kwargs
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences_train = sentences_train[:limit]
            y_train = y_train[:limit]
            sentences_test = sentences_test[:limit]
            y_test = y_test[:limit]
        self.sentences_train = sentences_train
        self.y_train = y_train
        self.sentences_test = sentences_test
        self.y_test = y_test
        self.args = kwargs['args']

        self.max_iter = max_iter

        if self.args.batch_size>0:
            self.batch_size = self.args.batch_size
        else:
            self.batch_size = batch_size

        if self.args.checkRobustness == True:
            self.checkRobustness = True
        else:
            self.checkRobustness = False

        if self.args.checkMultiLinguality == True:
            self.checkMultiLinguality = True
        else:
            self.checkMultiLinguality = False

        if self.args.robustnessSamples>0:
            self.robustnessSamples = self.args.robustnessSamples
        else:
            self.robustnessSamples = robustnessSamples

    def __call__(self, model, test_cache=None):
        print('use logRegClassificationEvaluator')
        scores = {}
        clf = LogisticRegression(
            random_state=self.seed,
            n_jobs=-1,
            max_iter=self.max_iter,
            verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
        )
        logger.info(f"Encoding {len(self.sentences_train)} training sentences...")


        if self.args.prompt:
            new_sentences = []
            print('with prompt')
            for s in self.sentences_train:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences_train = new_sentences

            new_sentences = []
            print('with prompt')
            for s in self.sentences_test:
                new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
            self.sentences_test = new_sentences

        if self.checkRobustness:
            if(self.args.task_name == 'EmotionClassification'):

                #vocabulary MFT
                editor = Editor()
                ret = editor.template("i can't stop feeling {mask} when i am alone and when i've got no {mask}")
                self.vocab_sentences_test = ret.data[:self.robustnessSamples]
                self.vocab_y_test = ([0]*self.robustnessSamples)
                
                ret = editor.template("i feel {mask} {mask} it got so bad")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("i {mask} a lot of time feeling dissapointed with myself for not doing a better job at {mask} my {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("i also {mask} lethargic {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("i have been feeling {mask} recently that i did not know back then that the abuse was not my {mask} and that it did not happen because of who i was")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                # # joy - 1

                ret = editor.template("i believe that feeling accepted in a {mask} way can be {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("i {mask} benevolent enough to buy them some {mask} and other treats")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("i giggle when i write this and shake my head in humbling {mask} but in a way i feel somewhat {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("i said before do feel {mask} to {mask} me this is something i am interested in finding out more about")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("i soon realized that an initial {mask} to an activity that feels playful is often followed by a desire to practice to perfect the talent that led to the {mask} enjoyment")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                # love - 2

                ret = editor.template("i {mask} we need a little romantic {mask} in the relationship")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("i feel the loving {mask} of my parents daily even though they have both been {mask} dead for almost {mask} now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("i could {mask} feel loving toward {mask} without them ever knowing it if i dont act like it")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("i suddenly feel that this is more than a {mask} love song that every {mask} could sing in front of their {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("i love what i do and i feel so {mask} and lucky to be able to travel and be creative and meet {mask} people and wake up every day loving my job")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                # anger - 3

                ret = editor.template("i get frustrated that {mask} issues from my past have had a {mask} negative effect on my behavior and feel {mask} must be angry that i have not resolved them by now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i {mask} more violent than ever right now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i feel more {mask} and annoyed by their {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i cannot {mask} why but i need to say please {mask} my {mask} i have heart and im not a heartless person")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("im {mask} cooped up and impatient and annoyingly {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                # fear - 4

                ret = editor.template("i can feel the pressure {mask} more so on my {mask} and im feeling slightly {mask} of myself which leads to unhappy thoughts not usually like my optimistic self i must say")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("i walked near the {mask} and i felt very obvious and uneasy all the warnings about {mask} crime i read in the guidebook and maybe some residual from years ago left me feeling {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("i {mask} unprotected even while {mask} alone")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("i often feel confused as to whether i have {mask} or just a really hard core sinful nature")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("i feel a restless {mask} heading our way")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                # surprise - 5

                ret = editor.template("as the {mask} slowly opened the audience gasped in surprise at the {mask} set design")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("a surprise party was planned in {mask} and her face lit up with {mask} as she entered the room")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("{mask} a present he couldn't hide his surprise when he saw the rare {mask} he had always wanted")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("hearing the news of a promotion at {mask} was an {mask} surprise that brought a smile to her face")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("the unexpected plot twist in the {mask} caught the audience off guard leaving them in {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                #Robustness Typo
                ret = Perturb.perturb(self.vocab_sentences_test, Perturb.add_typos, nsamples=len(self.vocab_sentences_test),keep_original=False)
                self.robust_sentences_test  = np.array(ret.data).reshape((len(self.vocab_sentences_test),))
                self.robust_y_test = self.vocab_y_test


                #Named Entity Recognition
                ret = editor.template('i could change the emphasis and say i am {first_name} and i m noticing i m feeling impatient', nsamples=self.robustnessSamples)
                self.ner_sentences_test = ret.data
                self.ner_y_test = [3]*self.robustnessSamples

                ret = editor.template('i looked at {first_name} this morning i named my left breast {first_name} my right one is {first_name} and i feel this weird mixture of anger and loss {first_name} wrote less than a month after her diagnosis', nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([4]*self.robustnessSamples)

                ret = editor.template('i get the impression that {first_name} was really feeling it but {first_name} still prefers her beloved {first_name} {last_name} purrrr', nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([2]*self.robustnessSamples)

                ret = editor.template('i get people asking me what it feels like to be the most hated man in {city} county said assessor {first_name} {last_name}', nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([0]*self.robustnessSamples)

                ret = editor.template('i was still feeling weepy and strung out so {first_name} treated me to ice cream and a movie a {movie}',movie=["top gun", "spiderman", "harry potter", "the adventures of huckeberry finn"], nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([0]*self.robustnessSamples)

                #negative tests

                try:
                    nlp = spacy.load('en_core_web_sm')
                    data = ['i could change the emphasis and say i am stella and i m noticing i m feeling impatient',
                            'i looked at mabel this morning i named my left breast mabel my right one is hazel and i feel this weird mixture of anger and loss valerie wrote less than a month after her diagnosis',
                            'i get the impression that banjo was really feeling it but molly still prefers her beloved katy perry purrrr',
                            'i get people asking me what it feels like to be the most hated man in dallas county said assessor steve helm',
                            'i was still feeling weepy and strung out so maggie treated me to ice cream and a movie a href http www',
                            "i feel embarrassed that it got so bad",
                            "i spend a lot of time feeling disappointed with myself for not doing a better job at attaining my goals",
                            "i also feel lethargic and again",
                            "i have been feeling regretful recently that i did not know back then that the abuse was not my fault and that it did not happen because of who i was but because of who they were",
                            "i believe that feeling accepted in a non judgemental way can be healing",
                            "i feel benevolent enough to buy them some peanuts and other treats",
                            "i write this i giggle and shake my head in humbling shame but in a way i feel somewhat triumphant",
                            "i said before do feel free to contact me this is something i am interested in finding out more about",
                            "i soon realized that an initial attraction to an activity that feels playful is often followed by a desire to practice to perfect the talent that led to the original enjoyment",
                            "i get frustrated that unresolved issues from my past have had a severe negative effect on my behavior and feel he must be angry that i have not resolved them by now",
                            "i feel more violent than ever right now"
                            ]
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.add_negation,keep_original=False)
                    self.neg_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.neg_y_test = [3,4,2,0,0,0,0,0,0,1,1,1,1,1,3,3]
                

                    data = ["i can't stop the anxiety i feel when i m alone when i ve got no distractions",
                            "i cannot explain why but i need to say please understand my feeling i have heart and im not a heartless person",
                            "i don't know if this helps at all but writing all of this has made me feel somewhat regretful of ashamed of who i was and while i have more to share i just don't think i can right now",
                            "i don't really believe because i walked through all the water stops in my first marathon and i actually don't think that walking is bad but dammit i was feeling stubborn and i wanted to get home and needed to be motivated by something",
                            "i don't want them to feel so pressured",
                            "i don't want to i feel irritated",
                            "i don't know about you but i'm feeling amp blessed",
                            "i don't have a schedule or childhood friends and feel a little timid about just getting out there by myself",
                            "i don't feel betrayed coz the backstabber had no grounds for their accusation but i'm just amazed at some people's ability to do such things",
                            "i eat a good breakfast i feel more energetic throughout the whole day and don't feel that o'clock slump"]

                    # ret = editor.template('i could change the emphasis and say i am {first_name} and i m noticing i m feeling impatient', nsamples=3)
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.remove_negation,keep_original=False)
                    self.pos_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.pos_y_test = [0,3,0,3,4,3,1,4,5,1]

                except:
                    nlp = spacy.load('en_core_web_sm')
                    data = ['i could change the emphasis and say i am stella and i m noticing i m feeling impatient',
                            'i looked at mabel this morning i named my left breast mabel my right one is hazel and i feel this weird mixture of anger and loss valerie wrote less than a month after her diagnosis',
                            'i get the impression that banjo was really feeling it but molly still prefers her beloved katy perry purrrr',
                            'i get people asking me what it feels like to be the most hated man in dallas county said assessor steve helm',
                            'i was still feeling weepy and strung out so maggie treated me to ice cream and a movie a href http www',
                            "i feel embarrassed that it got so bad",
                            "i spend a lot of time feeling disappointed with myself for not doing a better job at attaining my goals",
                            "i also feel lethargic and again",
                            "i have been feeling regretful recently that i did not know back then that the abuse was not my fault and that it did not happen because of who i was but because of who they were",
                            "i believe that feeling accepted in a non judgemental way can be healing",
                            "i feel benevolent enough to buy them some peanuts and other treats",
                            "i write this i giggle and shake my head in humbling shame but in a way i feel somewhat triumphant",
                            "i said before do feel free to contact me this is something i am interested in finding out more about",
                            "i soon realized that an initial attraction to an activity that feels playful is often followed by a desire to practice to perfect the talent that led to the original enjoyment",
                            "i get frustrated that unresolved issues from my past have had a severe negative effect on my behavior and feel he must be angry that i have not resolved them by now",
                            "i feel more violent than ever right now"
                            ]
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.add_negation,keep_original=False)
                    self.neg_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.neg_y_test = [3,4,2,0,0,0,0,0,0,1,1,1,1,1,3,3]
                

                    data = ["i can't stop the anxiety i feel when i m alone when i ve got no distractions",
                            "i cannot explain why but i need to say please understand my feeling i have heart and im not a heartless person",
                            "i don't know if this helps at all but writing all of this has made me feel somewhat regretful of ashamed of who i was and while i have more to share i just don't think i can right now",
                            "i don't really believe because i walked through all the water stops in my first marathon and i actually don't think that walking is bad but dammit i was feeling stubborn and i wanted to get home and needed to be motivated by something",
                            "i don't want them to feel so pressured",
                            "i don't want to i feel irritated",
                            "i don't know about you but i'm feeling amp blessed",
                            "i don't have a schedule or childhood friends and feel a little timid about just getting out there by myself",
                            "i don't feel betrayed coz the backstabber had no grounds for their accusation but i'm just amazed at some people's ability to do such things",
                            "i eat a good breakfast i feel more energetic throughout the whole day and don't feel that o'clock slump"]
                    
                    ret = editor.template('i could change the emphasis and say i am {first_name} and i m noticing i m feeling impatient', nsamples=3)
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.remove_negation,keep_original=False)
                    self.pos_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.pos_y_test = [0,3,0,3,4,3,1,4,5,1]

            elif(self.args.task_name == 'MTOPDomainClassification'):

                #vocabulary MFT
                editor = Editor()
                ret = editor.template("send {mask} via Facebook that {mask} to you later")
                self.vocab_sentences_test = ret.data[:self.robustnessSamples]
                self.vocab_y_test = ([0]*self.robustnessSamples)
                
                ret = editor.template("i {mask} you to start recording a {mask} message for Mohamed")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("{mask} send that message to my {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("Message Joe to say {mask} will be there in an {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("Text Josh to let him know {mask} bringing {mask} salad to his party next weekend.")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([0]*self.robustnessSamples)

                # # joy - 1

                ret = editor.template("I {mask} to call {mask} right now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("please show {mask} calls from mary from last {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("call {mask} granddaughter")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("Add {mask} to phonecall")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                ret = editor.template("Can you {mask} to see who is calling me {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([1]*self.robustnessSamples)

                # event - 2

                ret = editor.template("{mask} events in downtown Baltimore this {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("Who's going to The {mask} Festival?")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("things to do in {mask} viejo tonight")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("Music festivals this month in {mask} {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                ret = editor.template("rsvp {mask} to the third event")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([2]*self.robustnessSamples)

                # timer - 3

                ret = editor.template("i get frustrated that {mask} issues from my past have had a {mask} negative effect on my behavior and feel {mask} must be angry that i have not resolved them by now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i {mask} more violent than ever right now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i feel more {mask} and annoyed by their {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("i cannot {mask} why but i need to say please {mask} my {mask} i have heart and im not a heartless person")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                ret = editor.template("im {mask} cooped up and impatient and annoyingly {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([3]*self.robustnessSamples)

                # fear - 4

                ret = editor.template("{mask} play Pink now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("play {mask} {mask} on iheartradio")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("Create a list of Kendrick Lamar's {mask} songs")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("Play some {mask} Rap music.")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                ret = editor.template("Who was playing {mask} in the last song?")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([4]*self.robustnessSamples)

                # weather - 5

                ret = editor.template("How many {mask} of rain will come down {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("what will be the temperature in {mask} in 2 hours")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("When is our {mask} chance for rain?")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("Weather on {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)

                ret = editor.template("Report the outside temperature in {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([5]*self.robustnessSamples)


                # alarm - 6

                ret = editor.template("Set an alarm for 9pm {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([6]*self.robustnessSamples)

                ret = editor.template("Set an alarm for {mask} minutes from now")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([6]*self.robustnessSamples)

                ret = editor.template("I would like to be {mask} at noon please")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([6]*self.robustnessSamples)

                ret = editor.template("please {mask} alarm for monday at 5pm")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([6]*self.robustnessSamples)

                ret = editor.template("I have to {mask} my alarm for every {mask} this month")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([6]*self.robustnessSamples)

                # people - 7

                ret = editor.template("Tell me who's working at {mask}.")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([7]*self.robustnessSamples)

                ret = editor.template("Who {mask} from Creighton")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([7]*self.robustnessSamples)

                ret = editor.template("show me who is friends with {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([7]*self.robustnessSamples)

                ret = editor.template("Where in {mask} does Sarah work?")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([7]*self.robustnessSamples)

                ret = editor.template("Show {mask} of Jolene")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([7]*self.robustnessSamples)

                
                 # reminders - 8

                ret = editor.template("Did I set a reminder for {mask} 6 {mask} appointment")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([8]*self.robustnessSamples)

                ret = editor.template("remind to pick up {mask} after work")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([8]*self.robustnessSamples)

                ret = editor.template("Delete my reminder to {mask} Liberty University")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([8]*self.robustnessSamples)

                ret = editor.template("Kindly remove all reminders for {mask} lunch meetings next week")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([8]*self.robustnessSamples)

                ret = editor.template("I need to be reminded on Tuesday to send {mask} to Joyce's daughter for the {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([8]*self.robustnessSamples)


                # recipes - 9

                ret = editor.template("What kind of meat do I use in {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([9]*self.robustnessSamples)

                ret = editor.template("Look up ingredients for a mango {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([9]*self.robustnessSamples)

                ret = editor.template("Is there {mask} in eggs")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([9]*self.robustnessSamples)

                ret = editor.template("What {mask} should I bake quiche at")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([9]*self.robustnessSamples)

                ret = editor.template("{mask} the ingredients of spaghetti sauce")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([9]*self.robustnessSamples)


                # news - 10

                ret = editor.template("What are the {mask} headlines for today")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([10]*self.robustnessSamples)

                ret = editor.template("Any recent news about the {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([10]*self.robustnessSamples)

                ret = editor.template("What did Congress {mask} on {mask}")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([10]*self.robustnessSamples)

                ret = editor.template("the {mask} news of lakers")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([10]*self.robustnessSamples)

                ret = editor.template("{mask} news about Hilary Clinton")
                self.vocab_sentences_test = self.vocab_sentences_test + ret.data[:self.robustnessSamples]
                self.vocab_y_test = self.vocab_y_test + ([10]*self.robustnessSamples)



                #Robustness Typo
                ret = Perturb.perturb(self.vocab_sentences_test, Perturb.add_typos, nsamples=len(self.vocab_sentences_test),keep_original=False)
                self.robust_sentences_test  = np.array(ret.data).reshape((len(self.vocab_sentences_test),))
                self.robust_y_test = self.vocab_y_test


                #Named Entity Recognition
                ret = editor.template('Please call {first_name} {last_name}', nsamples=self.robustnessSamples)
                self.ner_sentences_test = ret.data
                self.ner_y_test = [1]*self.robustnessSamples

                ret = editor.template("Please send {first_name} a message that says 'are you going tonight?'", nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([0]*self.robustnessSamples)

                ret = editor.template("What's going on in {city} this weekend", nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([2]*self.robustnessSamples)

                ret = editor.template('set a reminder tomorrow for {first_name} to call {first_name} about a ride to the pool on saturday', nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([8]*self.robustnessSamples)

                ret = editor.template("Tell me about {first_name}'s friend, {first_name}", nsamples=self.robustnessSamples)
                self.ner_sentences_test = self.ner_sentences_test + ret.data
                self.ner_y_test = self.ner_y_test + ([7]*self.robustnessSamples)

                #negative tests

                try:
                    nlp = spacy.load('en_core_web_sm')
                    data = ['Do I have any friends that used to live in New Mexico?',
                            'I would like to be awoken at noon please',
                            "Tell me the calorie count for home made ice cream please",
                            "Please send Andrea a message that says 'are you going tonight?'",
                            "I need to call Dad right now",
                            "make one timer for eight seconds and another for 12 hours and another for 8 minutes",
                            "What concerts are happening in Phoenix this weekend",
                            "play nickelback for me",
                            "Is it going to rain around noon on Saturday?",
                            "please make alarm for monday at 5pm"]
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.add_negation,keep_original=False)
                    self.neg_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.neg_y_test = [7,6,9,0,1,3,2,4,5,6]
                

                    data = ["dont answer the phone",
                            "I dont want to miss the episode of love and hip , can you remind me at 5 pm.",
                            "Tell Sarah we won't be there",
                            "I won't be able to attend Church on Sunday so delete my reminder about that",
                            "Don't take any calls today.",
                            "I don't like this song",
                            "don't repeat timer",
                            "What can I do tonight that doesn't involve drinking",
                            "remove 12 hours from You aren't done running till this goes off timer",
                            "tell Kristin I am not feeling well, need to postpone our lunch"]

                    # ret = editor.template('i could change the emphasis and say i am {first_name} and i m noticing i m feeling impatient', nsamples=3)
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.remove_negation,keep_original=False)
                    self.pos_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.pos_y_test = [1,8,0,8,1,4,3,2,3,0]

                except:
                    nlp = spacy.load('en_core_web_sm')
                    data = ['Do I have any friends that used to live in New Mexico?',
                            'I would like to be awoken at noon please',
                            "Tell me the calorie count for home made ice cream please",
                            "Please send Andrea a message that says 'are you going tonight?'",
                            "I need to call Dad right now",
                            "make one timer for eight seconds and another for 12 hours and another for 8 minutes",
                            "What concerts are happening in Phoenix this weekend",
                            "play nickelback for me",
                            "Is it going to rain around noon on Saturday?",
                            "please make alarm for monday at 5pm"]
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.add_negation,keep_original=False)
                    self.neg_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.neg_y_test = [7,6,9,0,1,3,2,4,5,6]
                

                    data = ["dont answer the phone",
                            "I dont want to miss the episode of love and hip , can you remind me at 5 pm.",
                            "Tell Sarah we won't be there",
                            "I won't be able to attend Church on Sunday so delete my reminder about that",
                            "Don't take any calls today.",
                            "I don't like this song",
                            "don't repeat timer",
                            "What can I do tonight that doesn't involve drinking",
                            "remove 12 hours from You aren't done running till this goes off timer",
                            "tell Kristin I am not feeling well, need to postpone our lunch"]

                    # ret = editor.template('i could change the emphasis and say i am {first_name} and i m noticing i m feeling impatient', nsamples=3)
                    pdata = list(nlp.pipe(data))
                    ret = Perturb.perturb(pdata, Perturb.remove_negation,keep_original=False)
                    self.pos_sentences_test =  np.array(ret.data).reshape((len(pdata),))
                    self.pos_y_test = [1,8,0,8,1,4,3,2,3,0]


        if self.checkMultiLinguality:
            if(self.args.task_name == 'EmotionClassification'):
                

                df = pd.read_json('emotion_hi.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_hi_sentences_test = new_sentences
                self.mtlgl_hi_y_test = labels


                df = pd.read_json('emotion_de.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_de_sentences_test = new_sentences
                self.mtlgl_de_y_test = labels


                df = pd.read_json('emotion_el.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_el_sentences_test = new_sentences
                self.mtlgl_el_y_test = labels


                df = pd.read_json('emotion_es.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_es_sentences_test = new_sentences
                self.mtlgl_es_y_test = labels



                df = pd.read_json('emotion_fr.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_fr_sentences_test = new_sentences
                self.mtlgl_fr_y_test = labels



                df = pd.read_json('emotion_sw.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_sw_sentences_test = new_sentences
                self.mtlgl_sw_y_test = labels


                df = pd.read_json('emotion_th.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_th_sentences_test = new_sentences
                self.mtlgl_th_y_test = labels
            
            
            elif(self.args.task_name == 'MTOPDomainClassification'):
                

                df = pd.read_json('mtop_domain_hi.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_hi_sentences_test = new_sentences
                self.mtlgl_hi_y_test = labels


                df = pd.read_json('mtop_domain_de.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_de_sentences_test = new_sentences
                self.mtlgl_de_y_test = labels


                df = pd.read_json('mtop_domain_el.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_el_sentences_test = new_sentences
                self.mtlgl_el_y_test = labels


                df = pd.read_json('mtop_domain_es.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_es_sentences_test = new_sentences
                self.mtlgl_es_y_test = labels



                df = pd.read_json('mtop_domain_fr.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_fr_sentences_test = new_sentences
                self.mtlgl_fr_y_test = labels



                df = pd.read_json('mtop_domain_sw.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_sw_sentences_test = new_sentences
                self.mtlgl_sw_y_test = labels


                df = pd.read_json('mtop_domain_th.json')
                texts = df.text.values
                labels = df.label.values
                
                new_sentences = []
                print('with prompt')
                for s in texts:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                self.mtlgl_th_sentences_test = new_sentences
                self.mtlgl_th_y_test = labels


        #Nidhi: training data embeddings
        X_train = np.asarray(model.encode(self.sentences_train, batch_size=self.batch_size))
        logger.info(f"Encoding {len(self.sentences_test)} test sentences...")
        # if test_cache is None:
        #Nidhi: test data embeddings
        
        logger.info("Fitting logistic regression classifier...")
        clf.fit(X_train, self.y_train)


        if self.checkRobustness:
            vocab_X_test = np.asarray(model.encode(self.vocab_sentences_test, batch_size=self.batch_size))
            vocab_test_cache = vocab_X_test
            vocab_y_pred = clf.predict(vocab_X_test)
            vocab_accuracy = accuracy_score(self.vocab_y_test, vocab_y_pred)
            vocab_f1 = f1_score(self.vocab_y_test, vocab_y_pred, average="macro")

            robust_X_test = np.asarray(model.encode(self.robust_sentences_test, batch_size=self.batch_size))
            robust_test_cache = robust_X_test
            robust_y_pred = clf.predict(robust_X_test)
            robust_accuracy = accuracy_score(self.robust_y_test, robust_y_pred)
            robust_f1 = f1_score(self.robust_y_test, robust_y_pred, average="macro")

            ner_X_test = np.asarray(model.encode(self.ner_sentences_test, batch_size=self.batch_size))
            ner_test_cache = ner_X_test
            ner_y_pred = clf.predict(ner_X_test)
            ner_accuracy = accuracy_score(self.ner_y_test, ner_y_pred)
            ner_f1 = f1_score(self.ner_y_test, ner_y_pred, average="macro")

            neg_X_test = np.asarray(model.encode(self.neg_sentences_test, batch_size=self.batch_size))
            neg_test_cache = neg_X_test
            neg_y_pred = clf.predict(neg_X_test)
            neg_accuracy = accuracy_score(self.neg_y_test, neg_y_pred)
            neg_f1 = f1_score(self.neg_y_test, neg_y_pred, average="macro")

            pos_X_test = np.asarray(model.encode(self.pos_sentences_test, batch_size=self.batch_size))
            pos_test_cache = pos_X_test
            pos_y_pred = clf.predict(pos_X_test)
            pos_accuracy = accuracy_score(self.pos_y_test, pos_y_pred)
            pos_f1 = f1_score(self.pos_y_test, pos_y_pred, average="macro")

            logger.info("Incorrect vocab sentences")
            stacked_sentences = np.hstack((np.array(self.vocab_sentences_test).reshape(-1,1),np.array(self.vocab_y_test).reshape(-1,1),vocab_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,1] != stacked_sentences[:,2])])
            
            logger.info("Incorrect robustness sentences")
            stacked_sentences = np.hstack((np.array(self.robust_sentences_test).reshape(-1,1),np.array(self.robust_y_test).reshape(-1,1),robust_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,1] != stacked_sentences[:,2])])

            logger.info("Incorrect ner sentences")
            stacked_sentences = np.hstack((np.array(self.ner_sentences_test).reshape(-1,1),np.array(self.ner_y_test).reshape(-1,1),ner_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,1] != stacked_sentences[:,2])])

            logger.info("Incorrect Negative sentences")
            stacked_sentences = np.hstack((np.array(self.neg_sentences_test).reshape(-1,1),np.array(self.neg_y_test).reshape(-1,1),neg_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,1] != stacked_sentences[:,2])])

            logger.info("Incorrect Positive sentences")
            stacked_sentences = np.hstack((np.array(self.pos_sentences_test).reshape(-1,1),np.array(self.pos_y_test).reshape(-1,1),pos_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,1] != stacked_sentences[:,2])])

            logger.info("Robusness Accuracies...")

            logger.info("Vocab")
            logger.info(vocab_accuracy)
            logger.info(vocab_f1)
            
            logger.info("Robusness")
            logger.info(robust_accuracy)
            logger.info(robust_f1)
            
            logger.info("NER")
            logger.info(ner_accuracy)
            logger.info(ner_f1)
            
            logger.info("negative")
            logger.info(neg_accuracy)
            logger.info(neg_f1)

            logger.info("positive")
            logger.info(pos_accuracy)
            logger.info(pos_f1)

        if self.checkMultiLinguality:
            
            hindi_X_test = np.asarray(model.encode(self.mtlgl_hi_sentences_test, batch_size=self.batch_size))
            hindi_test_cache = hindi_X_test
            hindi_y_pred = clf.predict(hindi_X_test)
            hindi_accuracy = accuracy_score(self.mtlgl_hi_y_test, hindi_y_pred)
            hindi_f1 = f1_score(self.mtlgl_hi_y_test, hindi_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_hi_sentences_test,np.array(self.mtlgl_hi_y_test).reshape(-1,1),hindi_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("Multilinguality Accuracies...")
            logger.info("Hindi")
            logger.info(hindi_accuracy)
            logger.info(hindi_f1)
            



            de_X_test = np.asarray(model.encode(self.mtlgl_de_sentences_test, batch_size=self.batch_size))
            de_test_cache = de_X_test
            de_y_pred = clf.predict(de_X_test)
            de_accuracy = accuracy_score(self.mtlgl_de_y_test, de_y_pred)
            de_f1 = f1_score(self.mtlgl_de_y_test, de_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_de_sentences_test,np.array(self.mtlgl_de_y_test).reshape(-1,1),de_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("German")
            logger.info(de_accuracy)
            logger.info(de_f1)


            el_X_test = np.asarray(model.encode(self.mtlgl_el_sentences_test, batch_size=self.batch_size))
            el_test_cache = el_X_test
            el_y_pred = clf.predict(el_X_test)
            el_accuracy = accuracy_score(self.mtlgl_el_y_test, el_y_pred)
            el_f1 = f1_score(self.mtlgl_el_y_test, el_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_el_sentences_test,np.array(self.mtlgl_el_y_test).reshape(-1,1),el_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("Greek")
            logger.info(el_accuracy)
            logger.info(el_f1)

            
            es_X_test = np.asarray(model.encode(self.mtlgl_es_sentences_test, batch_size=self.batch_size))
            es_test_cache = es_X_test
            es_y_pred = clf.predict(es_X_test)
            es_accuracy = accuracy_score(self.mtlgl_es_y_test, es_y_pred)
            es_f1= f1_score(self.mtlgl_es_y_test, es_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_es_sentences_test,np.array(self.mtlgl_es_y_test).reshape(-1,1),es_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("Spanish")
            logger.info(es_accuracy)
            logger.info(es_f1)


            fr_X_test = np.asarray(model.encode(self.mtlgl_fr_sentences_test, batch_size=self.batch_size))
            fr_test_cache = fr_X_test
            fr_y_pred = clf.predict(fr_X_test)
            fr_accuracy = accuracy_score(self.mtlgl_fr_y_test, fr_y_pred)
            fr_f1= f1_score(self.mtlgl_fr_y_test, fr_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_fr_sentences_test,np.array(self.mtlgl_fr_y_test).reshape(-1,1),fr_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("French")
            logger.info(fr_accuracy)
            logger.info(fr_f1)


            sw_X_test = np.asarray(model.encode(self.mtlgl_sw_sentences_test, batch_size=self.batch_size))
            sw_test_cache = sw_X_test
            sw_y_pred = clf.predict(sw_X_test)
            sw_accuracy = accuracy_score(self.mtlgl_sw_y_test, sw_y_pred)
            sw_f1= f1_score(self.mtlgl_sw_y_test, sw_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_sw_sentences_test,np.array(self.mtlgl_sw_y_test).reshape(-1,1),sw_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("Swahili")
            logger.info(sw_accuracy)
            logger.info(sw_f1)

            th_X_test = np.asarray(model.encode(self.mtlgl_th_sentences_test, batch_size=self.batch_size))
            th_test_cache = th_X_test
            th_y_pred = clf.predict(th_X_test)
            th_accuracy = accuracy_score(self.mtlgl_th_y_test, th_y_pred)
            th_f1= f1_score(self.mtlgl_th_y_test, th_y_pred, average="macro")
            stacked_sentences = np.hstack((self.mtlgl_th_sentences_test,np.array(self.mtlgl_th_y_test).reshape(-1,1),th_y_pred.reshape(-1,1)))
            logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
            logger.info("Thai")
            logger.info(th_accuracy)
            logger.info(th_f1)



        X_test = np.asarray(model.encode(self.sentences_test, batch_size=self.batch_size))
        test_cache = X_test
        # else:
        #     X_test = test_cache
       
        logger.info("Evaluating...")
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        scores["accuracy"] = accuracy
        scores["f1"] = f1
        #Added by Nidhi
    
        stacked_sentences = np.hstack((self.sentences_test,np.array(self.y_test).reshape(-1,1),y_pred.reshape(-1,1)))
        logger.info(stacked_sentences[np.where(stacked_sentences[:,3] != stacked_sentences[:,4])])
        #np.concatenate self.sentences_test[:, 3] = y_pred
            

        # if binary classification
        if len(np.unique(self.y_train)) == 2:
            ap = average_precision_score(self.y_test, y_pred)
            scores["ap"] = ap

        return scores, test_cache
