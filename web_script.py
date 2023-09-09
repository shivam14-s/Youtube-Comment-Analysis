import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 
from googleapiclient.discovery import build
import nltk
import matplotlib.pyplot as plt
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

nltk.download("vader_lexicon", quiet=True)

class YoutubeComments:
    def __init__(self,video_url):
        api_key = 'AIzaSyAVafFo2t7DeOwDVlu508ckCJlCyTGStYo'
        self.video_url = video_url
        self.video_id = self.extract_video_id(video_url)
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    def extract_video_id(self,url) -> str:
        match = re.search(r'(?<=v=)[^&#]+', url)
        if match:
            return match.group(0)
        else:
            return None
    def get_video_title(self):
        """Get the title of the video
        :rtype: str
        :returns: Video title
        """

        response = self.youtube.videos().list(part="snippet", id=self.video_id).execute()

        return response["items"][0]["snippet"]["title"]
    def get_total_comments(self):
        response = self.youtube.videos().list(
            part='statistics',
            id=self.video_id
        ).execute()

        total_comments = response['items'][0]['statistics']['commentCount']
        return int(total_comments)
    def extract_comments(self, include_replies=False):
        x = 100
        if(include_replies):
            x = 10
        comments = []
        next_page_token = ''
        while True:
            # Retrieve the next page of comments
            response = self.youtube.commentThreads().list(
                part='snippet',
                videoId=self.video_id,
                pageToken=next_page_token,
                maxResults=x
            ).execute()

            # Extract the comments and metadata from the response
            for item in response['items']:
                comment_id = item['snippet']['topLevelComment']['id']
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                author = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
                published_date = item['snippet']['topLevelComment']['snippet']['publishedAt']
                comments.append({'comment': comment_text, 'author': author, 'published_date': published_date})

                # If include_replies is True, retrieve comment replies
                if include_replies:
                    reply_response = self.youtube.comments().list(
                        part='snippet',
                        parentId=comment_id,
                        maxResults=5
                    ).execute()
                    for reply_item in reply_response['items']:
                        reply_text = reply_item['snippet']['textDisplay']
                        reply_author = reply_item['snippet']['authorDisplayName']
                        reply_date = reply_item['snippet']['publishedAt']
                        comments.append({'comment': reply_text, 'author': reply_author, 'published_date': reply_date})

            # Check if there are more comments to retrieve
            if 'nextPageToken' in response:
                next_page_token = response['nextPageToken']
            else:
                break
        return comments
class Analyze:
    def __init__(self,df):
        self.df = df
    def comments_analyzed(self):
        return self.df.shape[0]
    def clean_comments(self):
        self.df["Cleaned Comment"] = (
            self.df["comment"]
            # remove whitespace
            .str.strip()
            # replace newlines with space
            .str.replace("\n", " ")
            # remove mentions and links
            .str.replace(r"(?:\@|http?\://|https?\://|www)\S+", "", regex=True)
            # remove punctuations, emojis, special characters
            .str.replace(r"[^\w\s]+", "", regex=True)
            # turn to lowercase
            .str.lower()
            # remove numbers
            .str.replace(r"\d+", "", regex=True)
            # remove hashtags
            .str.replace(r"#\S+", " ", regex=True)
        )
        # remove stop words
        stop_words = stopwords.words("english")
        self.df["Cleaned Comment"] = self.df["Cleaned Comment"].apply(
            lambda comment: " ".join([word for word in comment.split() if word not in stop_words])
        )
    def polarity_score(self,analyzer, text):
        scores = analyzer.polarity_scores(text)
        return scores["compound"]
    def score_to_sentiment(self,score):
        sentiment = ""
        if score <= -0.5:
            sentiment = "Negative"
        elif -0.5 < score <= 0.5:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"
        return sentiment
    def analyze_comments(self):
        analyzer = SentimentIntensityAnalyzer()
        self.df["Sentiment Score"] = self.df["Cleaned Comment"].apply(lambda comment: self.polarity_score(analyzer, comment))
        self.df["Sentiment"] = self.df["Sentiment Score"].apply(lambda score: self.score_to_sentiment(score))
    
    def create_pie_chart(self,video_title, filename) -> None:
        sentiment_counts_df = self.df["Sentiment"].value_counts().to_frame()
        sentiment_counts_df.reset_index(inplace=True)
        sentiment_counts_df.rename(columns={"index": "Sentiment", "count": "Counts"}, inplace=True)
        sentiment_counts_df = sentiment_counts_df.set_index("Sentiment")

        axis = sentiment_counts_df.plot.pie(
            y="Counts",
            ylabel="",
            figsize=(12, 12),
            fontsize=15,
            autopct="%1.1f%%",
            startangle=90,
            legend=False,
            textprops={"color": "w", "weight": "bold"},
        )

        plt.axis("equal")

        axis.set_facecolor("black")
        axis.set_title(
            f"Sentiment Analysis Results\n{video_title}",
            fontdict={"color": "white", "fontweight": "bold", "fontsize": 16},
            linespacing=2,
            pad=30,
        )
        axis.legend(labels=sentiment_counts_df.index, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        figure = axis.get_figure()
        figure.savefig(filename, facecolor="black", dpi=600) 
        return sentiment_counts_df
# Function to analyze comments and generate the pie chart
def analyze_com(video_url, include_replies):
    file_name = './piechart.jpg'
    obj1 = YoutubeComments(video_url)
    df = pd.DataFrame(obj1.extract_comments(include_replies))
    obj2 = Analyze(df)
    analyzed_comments = obj2.comments_analyzed()
    title =  obj1.get_video_title()
    obj2.clean_comments()
    obj2.analyze_comments()
    sentiment_counts_df = obj2.create_pie_chart(title,file_name)
    comment_num = obj1.get_total_comments()
    # Return the path to the saved pie chart file
    return file_name,title,comment_num,analyzed_comments,sentiment_counts_df
    # return file_name,title,comment_num,analyzed_comments
# Main web app code
def main():
    # Set up the web app interface
    st.title("YouTube Comments Sentiment Analyzer")
    st.write("Enter the URL of the YouTube video you want to analyze:")

    # Get user input for video URL
    video_url = st.text_input("Video URL")

    # Get user input for including comment replies
    include_replies = st.checkbox("Include comment replies")

    # Analyze comments and generate the pie chart
    if st.button("Analyze"):
        if video_url:
            # Call the analyze_comments function
            pie_chart_file,video_title,total_comments,analyzed_comments,sentiment_counts_df= analyze_com(video_url, include_replies)
            # pie_chart_file,video_title,total_comments,analyzed_comments= analyze_com(video_url, include_replies)
            # Display the total number of comments
            st.write(f"Total Comments: {total_comments}")
            st.write(f"Analyzed comments: {analyzed_comments}")
            st.write(f"Video Title: {video_title}")
            
            # Display the pie chart in the web app
            if pie_chart_file:
                st.write("Sentiment Analysis Results:")
                st.image(pie_chart_file)
                for index, row in sentiment_counts_df.iterrows():
                    sentiment = index
                    count = row["Counts"]
                    st.write(f"{sentiment}: {count} comments")
        else:
            st.write("Please enter a valid YouTube video URL.")

# Run the web app
if __name__ == "__main__":
    main()
