catalog = [
    ("cats", "cattwoman", 1998),
    ("alley cats", "cattwoman", 2001),    # ← second cattwoman book
    ("bats", "batman", 1997),
    ("spiders", "spiderman", 1999),
    ("ticks", "the tick", 2000),
]

borrowed ={"cats", "bats"}

def title_in_catalog(x):
    for title,author,year in catalog:
        if x == title:
            return True
    return False

def show_book_at(x):
    return catalog[x]

def books_by(x):
    return books_by_author[x]

def is_borrowed(x):
    return x in borrowed

def is_available(x):
    return title_in_catalog(x) and not is_borrowed(x)

books_by_author = {}
for title,author,year in catalog:
    if author not in books_by_author:
        books_by_author[author] = []
    books_by_author[author].append(title)


print(show_book_at(0))
print(books_by("cattwoman"))
print(is_borrowed("bats"))
print(is_available("ticks"))
