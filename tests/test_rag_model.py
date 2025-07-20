from core import RAGModel


# positive cases
def test_reading_local_directory():
    rag_model = RAGModel().local_read_dir("local_repo")
    assert True



def test_spliting_documents():
    try:
        rag_model = RAGModel().local_read_dir("local_repo").split_documents()
        assert True
    except:
        assert False    

 
 
# negative cases
def test_reading_local_directory_neg():
    try:
        rag_model = RAGModel().local_read_dir("local_repo")
        assert True
    except:
        assert False