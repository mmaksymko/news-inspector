import csv
import scrapetube

channel_usernames = []

titles = [
    (video['title']['runs'][0]['text'], 1)
    for username in channel_usernames
    for video in scrapetube.get_channel(channel_username=username, limit=10_000)
]


with open('clickbait_titles.csv', 'w', newline='', encoding='utf-8') as f:
    csv_writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_ALL, escapechar='\\')
    csv_writer.writerows(titles)
