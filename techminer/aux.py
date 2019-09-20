"""
"""
def cut_text(w):
    if isinstance(w, (int, float)):
        return w
    return w if len(w) < 35 else w[:31] + '... ' + w[w.find('['):]