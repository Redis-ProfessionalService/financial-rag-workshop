import yaml

def format_docs(docs):
    clean_docs = []
    for doc in docs:
        clean_docs.append(str(doc.page_content).replace("\n", " "))
    return clean_docs

def create_langchain_schemas_from_redis_schema(redis_yschema):
    with open(redis_yschema, "r") as yschema:
        schema = yaml.load(yschema, Loader=yaml.Loader)

    vector_schema = {
        "name": None,
        "algorithm": None,
        "dims": None,
        "distance_metric": None,
        "datatype": None,
    }

    index_schema = {
        "vector": [],
        "text": [],
        "tag": [],
        "numeric": [],
        "content_vector_key": None  # name of the vector field in langchain
    }

    for f in schema['fields']:
        # print(f'f={f}')
        if f['type'] == 'tag':
            index_schema['tag'] = index_schema['tag'] + [{'name': f['name']}]
        elif f['type'] == 'numeric':
            index_schema['numeric'] = index_schema['numeric'] + [{'name': f['name']}]
        elif f['type'] == 'text':
            index_schema['text'] = index_schema['text'] + [{'name': f['name']}]
        # TODO: add and test rest of the index field types
        elif f['type'] == 'vector':
            vector_schema = {
                "name": f['name'],
                "algorithm": str(f['attrs']['algorithm']).upper(),
                "dims": int(f['attrs']['dims']),
                "distance_metric": str(f['attrs']['distance_metric']).upper(),
                "datatype": str(f['attrs']['type']).upper(),
            }
            index_schema['vector'] = index_schema['vector'] + [vector_schema]
            index_schema['content_vector_key'] = f['name']  # langchain only accepts one vector name?
    return vector_schema, index_schema
