import os
import requests
import httplib2
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import tabula
import camelot as cam
# url = "https://maya.tase.co.il/reports/finance"
output_dir = ".\Outputs"
url = "https://maya.tase.co.il/en/reports/finance"
# links = []
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "http://google.com",
           "DNT":"1"
          }


#TODO rn headless (so it wont open the borwser)
def get_data_links():
    """
    get the links for the pdf file of the companies
    :return: a dict where company_pdf_links is a dict with key: company name,
                                                           val: the links for the website with the finance files of the key company
    """
    company_pdf_links, links = {}, []
    browser = webdriver.Chrome()
    browser.get(url)
    elem_list = browser.find_element(By.CSS_SELECTOR, " div.min-height.ng-scope > div > div > div > div > div")
    # print(elem_list)
    items = elem_list.find_elements(By.XPATH, '//div[@role="row"]')[1:] # delete the first row which are the cols titles
    for item in items:
        title = item.find_element(By.TAG_NAME, 'a').text
        # print("name of company is: ", title, "\n")
        links = [i.get_attribute('href') for i in item.find_elements(By.TAG_NAME, 'a') if "/2/" in i.get_attribute('href')]
        company_pdf_links[title] = set(links)
        # to avoid duplication of links yet still get all the necessary urls with pdf in them (can be more then 1)
        # print(company_pdf_links[title])
    # browser.quit()
    print("company_pdf_links are: \n", company_pdf_links)
    return company_pdf_links

def get_pdf_links(companies_data_links, pdf_links_list):
    """
    extract and filter only the pdf links of the company from the data link
    :param companies_data_links: the urls for the companies data
    :param pdf_links_list: a pointer to a list which we save the pdf files links
    :return: nothing, only update the given pdf_links_list by using its pointer
    """
    browser = webdriver.Chrome()
    pdf_links = []
    counter = 0
    while counter < len(companies_data_links):
        browser.get(companies_data_links[counter])
        elem = browser.find_elements(By.TAG_NAME, 'a')
        unfilterd_links = [i.get_attribute('href')for i in elem]
        filtered_links = list(filter(lambda  link: link is not None and
                                     len(link) != 0
                                     and link.endswith('.pdf'), unfilterd_links))

        pdf_links_list += filtered_links
        counter += 1

    # browser.quit()

def download_pdf(pdf_urls):
    """
    download the pdf file from a given url
    :param pdf_urls: the url of pdf to download
    """
    for url in pdf_urls:
        response = requests.get(url)
        if response.status_code == 200:  # 200 is good, 404 is bad
            file_path = os.path.join(output_dir, os.path.basename(url))
            # join the output_dir name with the tail of the url which is the pdf file name
            with open(file_path, 'wb') as f:
                f.write(response.content)

    # TODO
    # if theres already file in the output dir it will send them as well, need to fix it

def runner(company_name = None):
    """
    run the program to find and downland the finances files of all or several companies
    :param company_name: None by default - download all files, else a list of companies name

    """
    pdf_urls = []
    company_pdf_links = get_data_links()
    counter = 0
    for company in company_pdf_links:
        if company_name is None and counter<10:
            # print(f"company name is: {company} \n pdf_web_links are: {company_pdf_links[company]}")
            get_pdf_links(list(company_pdf_links[company]), pdf_urls)
            counter += 1
        if company_name is not None and company in company_name and counter<8:
            get_pdf_links(list(company_pdf_links[company]), pdf_urls)
            counter += 1
    print(pdf_urls)
    download_pdf(pdf_urls)


if __name__ == '__main__':
    runner()
    # runner(["ויתניה", "אנלייבקס", "tool"])








