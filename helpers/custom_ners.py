from spacy.lang.en import English
from helpers.ingestion import load_json_metadata
from helpers.ingestion import get_sec_data

# a simple toy rule-based NER just to demonstrate the use-case
def build_custom_sec_ner():
    sec_data = get_sec_data()
    patterns = []

    def get_patterns(ent):
        pattern_list = []
        ent_parts = str(ent).strip().split(" ")
        for part in ent_parts:
            pattern_list.append({"LOWER": str(part).lower()})
        return pattern_list

    for ticker in list(sec_data.keys()):
        if len(sec_data[ticker]["metadata_file"]) > 0:
            metadata = load_json_metadata(sec_data[ticker]["metadata_file"][0])
            patterns.append({"label": "ticker", "pattern": get_patterns(metadata['ticker']), "id": metadata['ticker']})
            patterns.append({"label": "company_name", "pattern": get_patterns(metadata['company_name']), "id": metadata['company_name']})
            patterns.append({"label": "sector", "pattern": get_patterns(metadata['sector']), "id": metadata['sector']})
            patterns.append({"label": "asset_class", "pattern": get_patterns(metadata['asset_class']), "id": metadata['asset_class']})
            patterns.append({"label": "exchange", "pattern": get_patterns(metadata['exchange']) , "id": metadata['exchange']})
        ruler.add_patterns(patterns)


nlp = English()
ruler = nlp.add_pipe("entity_ruler", config={"validate": True})
build_custom_sec_ner()


def get_entities(query):
    doc = nlp(query)
    entities = []
    for ent in doc.ents:
        entities.append({
            "label": ent.label_,
            "text": ent.text,
            "id": ent.ent_id_
        })
    return entities

# Simple OR based query translation based on rudimentary NLP pre-processing
def get_redis_filters(query):
    filters = []
    for ent in get_entities(query):
        value = "{" + f"{ent['id']}" + "}"
        filters.append(f"@{ent['label']}:{value}")
    if len(filters) == 0:
        return None
    elif len(filters) == 1:
        return filters[0]
    elif len(filters) > 1:
        return " | ".join(filters)
    else:
        return None
