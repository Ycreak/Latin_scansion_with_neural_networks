from lsnn.landing_zone.hypotactic.harvester import Harvester
import lsnn.utilities as util

poems: list = [                                                                                                       
    "calpurnius",                                                                                                     
    "catullus",                                                                                                       
    "ciceroaratea",                                                                                                   
    "columella10",                                                                                                    
    "ennius",                                                                                                         
    "germanicus",                                                                                                     
    "grattius",                                                                                                       
    # Horace                                                                                                          
    "odes1",                                                                                                          
    "odes2",                                                                                                          
    "odes3",                                                                                                          
    "odes4",                                                                                                          
    "epodes",                                                                                                         
    "sermones1",                                                                                                      
    "sermones2",                                                                                                      
    "epistulae1",                                                                                                     
    "epistulae2",                                                                                                     
    "arspoetica",                                                                                                     
    "martial1",                                                                                                       
    "martial2",                                                                                                       
    "martial3",                                                                                                       
    "martial4",                                                                                                       
    "martial5",                                                                                                       
    "martial6",                                                                                                       
    "martial7",                                                                                                       
    "martial8",                                                                                                       
    "martial9",                                                                                                       
    "martial10",                                                                                                      
    "martial11",                                                                                                      
    "martial12",                                                                                                      
    "martial13",                                                                                                      
    "martial14",                                                                                                      
    "spectaculis",                                                                                                    
    "persius",                                                                                                        
    "petronius2",               
    "phaedrus1",                                                                                                      
    "phaedrus2",                                                                                                      
    "phaedrus3",                                                                                                      
    "phaedrus4",                                                                                                      
    "phaedrus5",                                                                                                      
    "phaedrusapp",                                                                                                    
    "priapea",                                                                                                        
] 

# Proze                                                                                                               
prose: list = [                                                                                                       
    # Seneca                                                                                                          
    "agamemnon",                                                                                                      
    "hercfurens",                                                                                                     
    "herculesoet",                                                                                                    
    "medea",                                                                                                          
    "octavia",                                                                                                        
    "oedipus",                                                                                                        
    "phaedra",                                                                                                        
    "phoenissae",                                                                                                     
    "thyestes",                                                                                                       
    "troades",                                                                                                        
    # Terence                                                                                                         
    "heauton",                                                                                                        
    "adelphoe",                                                                                                       
    "andria",                                                                                                         
    "eunuchus",                                                                                                       
    "hecyra",                                                                                                         
    "phormio",                                                                                                        
    # Plautus                                                                                                         
    "amphitruo",                                                                                                      
    "asinaria",                                                                                                       
    "aulularia",                                                                                                      
    "bacchides",                                                                                                      
    "captivi",                                                                                                        
    "casina",                                                                                                         
    "cistellaria",                                                                                                    
    "curculio",                                                                                                       
    "epidicus",                                                                                                       
    "menaechmi",                                                                                                      
    "mercator",                                                                                                       
    "miles",                                                                                                          
    "mostellaria",                                                                                                    
    "persa",                                                                                                          
    "poenulus",                                                                                                       
    "pseudolus",              
    "rudens",                                                                                                         
    "stichus",                                                                                                        
    "trinummus",                                                                                                      
    "truculentus",                                                                                                    
]  

class Hypotactic:
    """
    Uses a webscraper to get the html source from the hypotactic website. Saves the plain text to the bucket.
    """
    def __init__(self) -> None:
        self.harvester = Harvester()

    def run(self, destination_path: str) -> None:
        # Harvest all poems
        for poem in poems:
            print(f'running harvester for {poem}')
            page = self.harvester.run(f'https://hypotactic.com/latin/index.html?Use_Id={poem}')
            util.write_pickle(f"{destination_path}/{poem}.html", page)

        # Harvest all texts
        for text in prose:
            print(f'running harvester for {text}')
            page = self.harvester.run(f'https://hypotactic.com/latin/index.html?Use_Id={text}')
            util.write_pickle(f"{destination_path}/{text}.html", page)
