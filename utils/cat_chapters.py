def categorize_chapters(count):
    if count < 20:
        return "<20"
    elif count < 50:
        return "<50"
    elif count < 100:
        return "50+"
    else:
        return "100+"