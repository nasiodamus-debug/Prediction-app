"""A tiny stand-in for a Playwright page that serves recorded HTML, so the REAL selectors in SELECTORS are exercised without
a browser or a network. It understands exactly what the bet-history selectors use: comma groups, descendant chains, tag
names and class names. Test helper only."""
from __future__ import annotations

from html.parser import HTMLParser

VOID_TAGS = {"br", "img", "input", "hr", "meta", "link", "i"}


class Node:
    def __init__(self, tag, classes, parent):
        self.tag, self.classes, self.parent = tag, set(classes), parent
        self.children: list = []   # Node or str

    def text(self) -> str:
        return "".join(c if isinstance(c, str) else c.text() for c in self.children)

    def descendants(self):
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.descendants()


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", (), None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        classes = (dict(attrs).get("class") or "").split()
        node = Node(tag, classes, self.cur)
        self.cur.children.append(node)
        if tag not in VOID_TAGS:
            self.cur = node

    def handle_endtag(self, tag):
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def parse(html: str) -> Node:
    builder = _Builder()
    builder.feed(html)
    return builder.root


def _compound(text: str):
    tag, _, rest = text.partition(".")
    return (tag or None), [c for c in rest.split(".") if c]


def _is(node: Node, compound) -> bool:
    tag, classes = compound
    return (tag is None or node.tag == tag) and all(c in node.classes for c in classes)


def _matches_chain(node: Node, chain) -> bool:
    if not _is(node, chain[-1]):
        return False
    if len(chain) == 1:
        return True
    up = node.parent
    while up is not None:
        if _matches_chain(up, chain[:-1]):
            return True
        up = up.parent
    return False


def select(scope: Node, selector: str) -> list[Node]:
    chains = [[_compound(part) for part in group.split()] for group in selector.split(",") if group.strip()]
    return [n for n in scope.descendants() if any(_matches_chain(n, chain) for chain in chains)]


class FakeLocator:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    async def count(self):
        return len(self.nodes)

    def nth(self, index):
        return FakeLocator([self.nodes[index]])

    @property
    def first(self):
        return FakeLocator(self.nodes[:1])

    async def inner_text(self):
        return self.nodes[0].text()

    def locator(self, selector):
        found = []
        for node in self.nodes:
            found += select(node, selector)
        return FakeLocator(found)


class FakeHistoryPage:
    def __init__(self, html: str, log: list):
        self.root, self.log, self.closed = parse(html), log, False

    async def goto(self, url, **kwargs):
        self.log.append(url)

    def locator(self, selector):
        return FakeLocator(select(self.root, selector))

    async def wait_for_timeout(self, ms):
        return None

    async def close(self):
        self.closed = True


class FakeHistoryContext:
    """context.new_page() gives a page that serves `html` (a string, or a function of the URL it is later sent to)."""

    def __init__(self, html: str):
        self.html, self.urls, self.pages = html, [], []

    async def new_page(self):
        page = FakeHistoryPage(self.html, self.urls)
        self.pages.append(page)
        return page
