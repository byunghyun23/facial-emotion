# pip install spotipy
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random


class Music:
    def __init__(self):
        # Set emotions
        self.emotions = {'기쁨': ['happy', 'delighted', 'glad', 'exiting', 'pleased',
                                'energetic', 'cheerful', 'satisfied', 'fulfilled', 'overjoyed'],
                         '슬픔': ['sad', 'unhappy', 'tearful', 'gloomy', 'depressed',
                                'dejected', 'heartbroken', 'sorrowful', 'hurt', 'disappointed'],
                         '분노': ['angry', 'mad', 'upset', 'furious', 'livid',
                                'irritated', 'enraged', 'incensed', 'outburst', 'resentful'],
                         '불안': ['anxious', 'nervous', 'uneasy', 'tense', 'worried',
                                'apprehensive', 'edgy', 'restless', 'uncertain', 'shaky'],
                         '상처': ['hurt', 'injured', 'damaged', 'wounded', 'scarred',
                                'sore', 'bruised', 'painful', 'aching', 'afflicted'],
                         '중립': ['neutral', 'emotionless', 'unfeeling', 'cold', 'unresponsive',
                                'impassive', 'apathetic', 'stoic', 'indifferent', 'unbiased']
                         }

        # Set up your Spotify API credentials
        self.client_id = 'YOUR_ID'
        self.client_secret = 'YOUR_SECRET'

        # Create a client credentials manager
        self.client_credentials_manager = SpotifyClientCredentials(self.client_id, self.client_secret)

        # Create a Spotipy client
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)

    def get_music(self, kor_emotion='기쁨', genre='k-pop ', gradio=False):
        recommend = ''

        print('kor_emotion', kor_emotion)
        # Search for tracks based on genre and mood
        eng_emotion = self.emotions[kor_emotion][0]
        selected_mood = random.sample(self.emotions[kor_emotion], len(self.emotions[kor_emotion]) // 2)
        mood = ' '.join(m for m in selected_mood)
        print('mood:', mood)

        query = genre + mood  # Replace with your desired mood
        results = self.sp.search(q=query, type='track')

        # Get the track names
        best_track_name = ''
        best_artist_name = ''
        pop = 0
        for track in results['tracks']['items']:
            track_name = track['name']
            artist_name = track['artists'][0]['name']
            popularity = track['popularity']
            # print(f"Track: {track_name}")
            # print(f"Artist: {artist_name}")
            # print(f"popularity: {popularity}")
            # print()

            if popularity > pop:
                pop = popularity
                best_track_name = track_name
                best_artist_name = artist_name

        if best_track_name != '':
            # print(f"Best Track: {best_track_name}")
            # print(f"Best Artist: {best_artist_name}")
            recommend += f'Are you {eng_emotion} today?\n'
            recommend += f"I'll recommend \"{best_track_name}\" by \"{best_artist_name}\"."
            if gradio is True:
                recommend += '\n\n'
                recommend += f'Link: https://www.youtube.com/results?search_query={best_artist_name.replace(" ", "+")}-{best_track_name.replace(" ", "+")}\n'
                # recommend += f'Click &lt;a href="https://www.youtube.com/results?search_query={best_artist_name.replace(" ", "+")}-{best_track_name.replace(" ", "+")}"&gt;here&lt;/a&gt;\n'
        else:
            recommend += "Sorry, I don't have any music to recommend today.\n"

        return recommend





