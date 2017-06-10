import os
import requests
import sys
import urllib
from bs4 import BeautifulSoup as bs

base_dir = "midi"

if __name__ == "__main__":
    page = requests.get(sys.argv[1])
    soup = bs(page.text, "html.parser")

    for link in soup.findAll("a"):
        link_href = link.get("href")
        if link.get("name"):
            game_name = link.get("name")
            game_path = os.path.join(base_dir, game_name)
            if not os.path.exists(game_path):
                os.makedirs(game_path)
                print(game_name)
        if link_href and link_href.endswith(".mid"):
            full_path = os.path.join(game_path, link_href)
            full_link = urllib.parse.urljoin(sys.argv[1], link_href)
            song = requests.get(full_link)
            with open(full_path, "wb") as f:
                f.write(song.content)
