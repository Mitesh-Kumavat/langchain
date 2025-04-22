from langchain.document_loaders import WebBaseLoader

url = "https://www.flipkart.com/noise-icon-2-1-8-display-bluetooth-calling-women-s-edition-ai-voice-assistant-smartwatch/p/itm968c523d99eae?pid=SMWGEH7VNGPYN5NV&lid=LSTSMWGEH7VNGPYN5NVEORYD7&marketplace=FLIPKART&store=ajy&srno=b_1_1&otracker=browse&fm=organic&iid=en_d3QcCFihH-kG_6bK2qXOLp0g_SDmo52KmoG4Cio5-WJ0YIcfYgcL0BOZt4BXbNHQK-txWLFXOsYDT6Vb6IDiQg%3D%3D&ppt=hp&ppn=homepage&ssid=rvfh3a5av40000001745321195571"

loader = WebBaseLoader(url)

# we can also pass multiple list of urls like this : 
# urls = ['https://flipkart.com', 'https://www.amazon.in']
# loader = WebBaseLoader(urls)

docs = loader.load()

print(docs[0].page_content)