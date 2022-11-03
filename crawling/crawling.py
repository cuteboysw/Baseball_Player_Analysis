from bs4 import BeautifulSoup
from urllib.request import urlopen

kbopage=urlopen('https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx')
soup = BeautifulSoup(kbopage, 'html.parser')

print(soup.div())