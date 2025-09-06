# -----------------------------
# Lexer
# -----------------------------

# Token definitions
# Each token is a tuple of (TYPE, REGEX)
# Order matters: more specific patterns should come before generic ones
TOKEN_REGEX = [
    ("NUMBER",   r"\d+"),                                # integers
    ("STRING",   r"\".*?\""),                            # double-quoted strings
    ("IDENT",    r"[A-Za-z_][A-Za-z0-9_]*"),             # identifiers/keywords
    ("OP",       r"[+\-*/<>]=?|==|\||&|\*"),             # operators (+, -, *, ==, etc.)
    ("SYMBOL",   r"[:=]|\.\.|,|\(|\)|\[|\]|\{|\}"),      # symbols and punctuation
    ("NEWLINE",  r"\n"),                                 # line breaks
    ("SKIP",     r"[ \t]+"),                             # whitespace
]

def tokenize(src: str):
    """
    Tokenizer for Stacker source code.

    Args:
        src (str): The source code string.

    Returns:
        list of (type, value) tuples.

    Raises:
        SyntaxError: if an unexpected character is found.
    """
    tokens = []
    pos = 0
    while pos < len(src):
        for ttype, pattern in TOKEN_REGEX:
            m = re.match(pattern, src[pos:])
            if m:
                # Ignore whitespace & newlines
                if ttype not in ("SKIP", "NEWLINE"):
                    tokens.append((ttype, m.group(0)))
                pos += len(m.group(0))
                break
        else:
            # No token matched
            raise SyntaxError(f"Unexpected char {src[pos]!r} at position {pos}")
    return tokens

# -----------------------------
# IR (Intermediate Representation)
# -----------------------------

IR = {
    # Stack operations
    "PUSH":   1,   # push a constant or immediate value
    "POP":    2,   # discard top of stack
    "DUP":    3,   # duplicate top of stack

    # Memory operations
    "LOAD":   4,   # load a variable from memory
    "STORE":  5,   # store top of stack into variable

    # Arithmetic (extendable)
    "ADD":    6,
    "SUB":    7,
    "MUL":    8,
    "DIV":    9,

    # Comparison
    "CMPEQ":  10,  # ==
    "CMPLT":  11,  # <
    "CMPGT":  12,  # >

    # Control flow
    "JMP":    13,  # jump unconditionally
    "JZ":     14,  # jump if zero (falsey)
    "JNZ":    15,  # jump if not zero (truthy)

    # I/O & termination
    "PRINT":  16,  # print top of stack
    "HALT":   17,  # stop execution
}

# Reverse lookup (for debugging & disassembly)
IR_NAMES = {v: k for k, v in IR.items()}

# -----------------------------
# AST (Abstract Syntax Tree)
# -----------------------------

# === Base Node ===
class Node:
    """Base class for all AST nodes."""
    def eval(self, env):
        raise NotImplementedError("All AST nodes must implement eval()")


# === Top-level structures ===
class Task(Node):
    """A named block of statements, like 'task main:'."""
    def __init__(self, name, body):
        self.name = name
        self.body = body

    def eval(self, env):
        for stmt in self.body:
            stmt.eval(env)
        return env


# === Statements ===
class VarDecl(Node):
    """Variable declaration/assignment: let x = expr"""
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def eval(self, env):
        env[self.name] = self.expr.eval(env)
        return env[self.name]


class Say(Node):
    """Print statement: say expr"""
    def __init__(self, expr):
        self.expr = expr

    def eval(self, env):
        val = self.expr.eval(env)
        print(val)
        return val


class Match(Node):
    """Pattern match statement: match expr: case ... end"""
    def __init__(self, expr, cases):
        self.expr = expr
        self.cases = cases

    def eval(self, env):
        value = self.expr.eval(env)
        for case in self.cases:
            local_env = dict(env)
            if case.pattern.match(value, local_env):
                for stmt in case.block:
                    stmt.eval(local_env)
                env.update(local_env)
                return True
        return False


class Chart(Node):
    """Chart visualization: chart expr [tree]"""
    def __init__(self, expr, tree=False):
        self.expr = expr
        self.tree = tree

    def eval(self, env):
        val = self.expr.eval(env)
        if self.tree:
            print("Beanstalk Chart (tree):")
            print_tree(val)
        else:
            print("Beanstalk Chart (dash):")
            if isinstance(val, list):
                line = ""
                for i, v in enumerate(val):
                    line += str(v)
                    if i < len(val) - 1:
                        line += "—"
                    print(line)
            else:
                print(val)
        return val


# === Expressions ===
class Push(Node):
    """Literal constant (numbers, strings)."""
    def __init__(self, value):
        self.value = value

    def eval(self, env):
        return self.value


class Var(Node):
    """Variable reference."""
    def __init__(self, name):
        self.name = name

    def eval(self, env):
        v = env[self.name]
        if callable(v):   # support lazy values
            v = v()
            env[self.name] = v
        return v


# -----------------------------
# Pattern AST (Executable)
# -----------------------------
class Pattern:
    """Base class for all patterns."""
    def match(self, value, env):
        raise NotImplementedError


class LiteralPattern(Pattern):
    """Match a literal constant (number, string)."""
    def __init__(self, value):
        self.value = value

    def match(self, value, env):
        return value == self.value


class WildcardPattern(Pattern):
    """Match anything: '_'."""
    def match(self, value, env):
        return True


class VarPattern(Pattern):
    """Bind a variable in a pattern: e.g., 'x'."""
    def __init__(self, name):
        self.name = name

    def match(self, value, env):
        env[self.name] = value
        return True


class TuplePattern(Pattern):
    """Tuple pattern: (a, b=1, c)."""
    def __init__(self, elts):
        # elts = list of (pattern, default_expr, lazy)
        self.elts = elts

    def match(self, value, env):
        if not isinstance(value, tuple):
            return False
        vals = list(value)
        vi = 0
        for sub, default, lazy in self.elts:
            if vi < len(vals):
                if not sub.match(vals[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    env[name] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True


class ListPattern(Pattern):
    """List pattern: [a, b=2, *rest]."""
    def __init__(self, elts):
        self.elts = elts  # list of (pattern, default_expr, lazy)

    def match(self, value, env):
        if not isinstance(value, list):
            return False
        vi = 0
        n = len(value)
        for i, (sub, default, lazy) in enumerate(self.elts):
            if isinstance(sub, SplatPattern):
                if sub.beanstalk:
                    for j in range(1, n + 1):
                        env[sub.var] = value[:j]
                        run_beanstalk_hooks(sub.var, env[sub.var])
                    return True
                env[sub.var] = value[vi:-len(self.elts[i + 1:])] if self.elts[i + 1:] else value[vi:]
                vi = n - len(self.elts[i + 1:])
            elif vi < n:
                if not sub.match(value[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    env[name] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True


class DictPattern(Pattern):
    """Dict pattern: {key: pat, name="default"}."""
    def __init__(self, entries):
        self.entries = entries  # list of (key, pattern, default_expr, lazy)

    def match(self, value, env):
        if not isinstance(value, dict):
            return False
        for k, sub, default, lazy in self.entries:
            if k in value:
                if not sub.match(value[k], env):
                    return False
            else:
                if default is not None:
                    env[k] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True


class SplatPattern(Pattern):
    """Splat pattern: *rest (optional beanstalk)."""
    def __init__(self, var, beanstalk=False):
        self.var = var
        self.beanstalk = beanstalk

    def match(self, value, env):
        if self.beanstalk and isinstance(value, list):
            for j in range(1, len(value) + 1):
                env[self.var] = value[:j]
                run_beanstalk_hooks(self.var, env[self.var])
            return True
        env[self.var] = value
        return True


# === Match Case ===
class Case(Node):
    """A single case in a match expression."""
    def __init__(self, pattern, block):
        self.pattern = pattern
        self.block = block

    def eval(self, env):
        # Case nodes are evaluated inside Match, not directly
        return None


# -----------------------------
# Pattern AST (Executable)
# -----------------------------

class Pattern:
    """Base class for all pattern types."""
    def match(self, value, env):
        raise NotImplementedError("Pattern subclasses must implement match()")


class LiteralPattern(Pattern):
    """Match a literal constant."""
    def __init__(self, value):
        self.value = value

    def match(self, value, env):
        return value == self.value


class WildcardPattern(Pattern):
    """Match anything (underscore)."""
    def match(self, value, env):
        return True


class VarPattern(Pattern):
    """Bind a variable in a pattern."""
    def __init__(self, name):
        self.name = name

    def match(self, value, env):
        env[self.name] = value
        return True


class TuplePattern(Pattern):
    """Match tuples, with optional defaults."""
    def __init__(self, elts):
        # elts = list of (pattern, default_expr, lazy_flag)
        self.elts = elts

    def match(self, value, env):
        if not isinstance(value, tuple):
            return False
        vals = list(value)
        vi = 0
        for sub, default, lazy in self.elts:
            if vi < len(vals):
                if not sub.match(vals[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    if lazy:
                        env[name] = lambda d=default, e=dict(env): gen_expr(d, e)
                    else:
                        env[name] = gen_expr(default, env)
                else:
                    return False
        return True


class ListPattern(Pattern):
    """Match lists, with defaults, splats, and optional beanstalking."""
    def __init__(self, elts):
        # elts = list of (pattern, default_expr, lazy_flag)
        self.elts = elts

    def match(self, value, env):
        if not isinstance(value, list):
            return False
        vi = 0
        n = len(value)
        for i, (sub, default, lazy) in enumerate(self.elts):
            if isinstance(sub, SplatPattern):
                if sub.beanstalk:
                    # contemplative beanstalking expansion
                    for j in range(1, n + 1):
                        env[sub.var] = value[:j]
                        run_beanstalk_hooks(sub.var, env[sub.var])
                    return True
                # normal splat: capture slice
                env[sub.var] = value[vi:-len(self.elts[i + 1:])] if self.elts[i + 1:] else value[vi:]
                vi = n - len(self.elts[i + 1:])
            elif vi < n:
                if not sub.match(value[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    if lazy:
                        env[name] = lambda d=default, e=dict(env): gen_expr(d, e)
                    else:
                        env[name] = gen_expr(default, env)
                else:
                    return False
        return True


class DictPattern(Pattern):
    """Match dictionaries with nested patterns and defaults."""
    def __init__(self, entries):
        # entries = list of (key, pattern, default_expr, lazy_flag)
        self.entries = entries

    def match(self, value, env):
        if not isinstance(value, dict):
            return False
        for k, sub, default, lazy in self.entries:
            if k in value:
                if not sub.match(value[k], env):
                    return False
            else:
                if default is not None:
                    if lazy:
                        env[k] = lambda d=default, e=dict(env): gen_expr(d, e)
                    else:
                        env[k] = gen_expr(default, env)
                else:
                    return False
        return True


class SplatPattern(Pattern):
    """Match a splat (remaining items). Can also act as beanstalk."""
    def __init__(self, var, beanstalk=False):
        self.var = var
        self.beanstalk = beanstalk

    def match(self, value, env):
        # Direct use outside lists/tuples is rare but supported
        if self.beanstalk and isinstance(value, list):
            for j in range(1, len(value) + 1):
                env[self.var] = value[:j]
                run_beanstalk_hooks(self.var, env[self.var])
            return True
        env[self.var] = value
        return True

# -----------------------------
# Parser
# -----------------------------

def parse(tokens):
    """
    Convert a flat token stream into a Stacker AST (list of Task nodes).
    """

    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else ("EOF", "EOF")

    def match(ttype, value=None):
        """
        Consume a token if it matches type (and optionally value).
        """
        nonlocal pos
        if pos < len(tokens) and tokens[pos][0] == ttype and (value is None or tokens[pos][1] == value):
            tok = tokens[pos]
            pos += 1
            return tok
        return None

    # -----------------------------
    # Expression parsing
    # -----------------------------
    def parse_expr():
        if peek()[0] == "NUMBER":
            return Push(int(match("NUMBER")[1]))
        if peek()[0] == "STRING":
            return Push(match("STRING")[1].strip('"'))
        if peek()[0] == "IDENT":
            return Var(match("IDENT")[1])
        raise SyntaxError(f"Unexpected expression token {peek()}")

    # -----------------------------
    # Pattern parsing
    # -----------------------------
    def parse_pattern():
        if match("IDENT", "_"):
            return WildcardPattern()

        if peek()[0] == "NUMBER":
            return LiteralPattern(int(match("NUMBER")[1]))
        if peek()[0] == "STRING":
            return LiteralPattern(match("STRING")[1].strip('"'))
        if peek()[0] == "IDENT":
            return VarPattern(match("IDENT")[1])

        # Tuple pattern
        if match("SYMBOL", "("):
            elts = []
            while not match("SYMBOL", ")"):
                sub = parse_pattern()
                default = None
                lazy = False
                if match("SYMBOL", "="):
                    default = parse_expr()
                    if match("IDENT", "lazy"):
                        lazy = True
                elts.append((sub, default, lazy))
                match("SYMBOL", ",")
            return TuplePattern(elts)

        # List pattern
        if match("SYMBOL", "["):
            elts = []
            while not match("SYMBOL", "]"):
                if match("OP", "*"):
                    var = match("IDENT")[1]
                    beanstalk = (var in ("bean", "beanstalk"))
                    elts.append((SplatPattern(var, beanstalk), None, False))
                else:
                    sub = parse_pattern()
                    default = None
                    lazy = False
                    if match("SYMBOL", "="):
                        default = parse_expr()
                        if match("IDENT", "lazy"):
                            lazy = True
                    elts.append((sub, default, lazy))
                match("SYMBOL", ",")
            return ListPattern(elts)

        # Dict pattern
        if match("SYMBOL", "{"):
            entries = []
            while not match("SYMBOL", "}"):
                key_tok = match("IDENT") or match("STRING")
                key = key_tok[1].strip('"')
                pat = VarPattern(key)
                default = None
                lazy = False
                if match("SYMBOL", ":"):
                    pat = parse_pattern()
                if match("SYMBOL", "="):
                    default = parse_expr()
                    if match("IDENT", "lazy"):
                        lazy = True
                entries.append((key, pat, default, lazy))
                match("SYMBOL", ",")
            return DictPattern(entries)

        raise SyntaxError(f"Unexpected pattern token {peek()}")

    # -----------------------------
    # Case parsing
    # -----------------------------
    def parse_case():
        pat = parse_pattern()
        match("SYMBOL", ":")
        blk = parse_block()
        return Case(pat, blk)

    # -----------------------------
    # Block parsing
    # -----------------------------
    def parse_block():
        block = []
        while not (peek()[0] == "IDENT" and peek()[1] in ("end", "case")):
            # say expr
            if match("IDENT", "say"):
                block.append(Say(parse_expr()))
                continue
            # chart expr [tree]
            if match("IDENT", "chart"):
                expr = parse_expr()
                tree = False
                if match("IDENT", "tree"):
                    tree = True
                block.append(Chart(expr, tree))
                continue
            # let name = expr
            if match("IDENT", "let"):
                name = match("IDENT")[1]
                match("SYMBOL", "=")
                block.append(VarDecl(name, parse_expr()))
                continue
            # match expr: case...
            if match("IDENT", "match"):
                expr = parse_expr()
                match("SYMBOL", ":")
                cases = []
                while match("IDENT", "case"):
                    cases.append(parse_case())
                match("IDENT", "end")
                block.append(Match(expr, cases))
                continue
            # fallback: allow raw expr (side effect only if used)
            block.append(parse_expr())
        return block

    # -----------------------------
    # Top-level (task parsing)
    # -----------------------------
    ast = []
    while pos < len(tokens):
        if match("IDENT", "task"):
            name = match("IDENT")[1]
            match("SYMBOL", ":")
            body = parse_block()
            match("IDENT", "end")
            ast.append(Task(name, body))
        else:
            pos += 1  # skip unknown
    return ast

# -----------------------------
# Pattern Matching Engine
# -----------------------------

def run_match(expr_node, cases, env):
    """
    Execute a match expression.
    - expr_node: AST expression whose value we match against.
    - cases: list of Case nodes.
    - env: environment (dict of vars).
    """
    value = expr_node.eval(env)
    matched = False

    for case in cases:
        # clone environment so each case gets a fresh scope
        local_env = dict(env)

        # try to match
        if case.pattern.match(value, local_env):
            # execute case body
            for stmt in case.block:
                stmt.eval(local_env)
            # update outer env with successful bindings
            env.update(local_env)
            matched = True
            break

    return matched

# -----------------------------
# Pattern Codegen (Executable)
# -----------------------------

def gen_pattern_code(pattern, value, env):
    """
    Apply pattern-matching codegen immediately.
    This function drives the .match() calls on patterns,
    and ensures defaults + lazy bindings are handled inline.

    Args:
        pattern: a Pattern node (LiteralPattern, ListPattern, etc.)
        value: the value to test against
        env: current environment (dict)

    Returns:
        bool — True if matched, False otherwise
    """

    # === Literal ===
    if isinstance(pattern, LiteralPattern):
        return value == pattern.value

    # === Wildcard ===
    elif isinstance(pattern, WildcardPattern):
        return True

    # === Variable Binding ===
    elif isinstance(pattern, VarPattern):
        env[pattern.name] = value
        return True

    # === Tuple ===
    elif isinstance(pattern, TuplePattern):
        if not isinstance(value, tuple):
            return False
        vals = list(value)
        vi = 0
        for sub, default, lazy in pattern.elts:
            if vi < len(vals):
                if not gen_pattern_code(sub, vals[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    env[name] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True

    # === List ===
    elif isinstance(pattern, ListPattern):
        if not isinstance(value, list):
            return False
        vi = 0
        n = len(value)
        for i, (sub, default, lazy) in enumerate(pattern.elts):
            if isinstance(sub, SplatPattern):
                if sub.beanstalk:
                    for j in range(1, n + 1):
                        env[sub.var] = value[:j]
                        run_beanstalk_hooks(sub.var, env[sub.var])
                    return True
                env[sub.var] = value[vi:-len(pattern.elts[i + 1:])] if pattern.elts[i + 1:] else value[vi:]
                vi = n - len(pattern.elts[i + 1:])
            elif vi < n:
                if not gen_pattern_code(sub, value[vi], env):
                    return False
                vi += 1
            else:
                if default is not None:
                    name = getattr(sub, "name", "_")
                    env[name] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True

    # === Dict ===
    elif isinstance(pattern, DictPattern):
        if not isinstance(value, dict):
            return False
        for k, sub, default, lazy in pattern.entries:
            if k in value:
                if not gen_pattern_code(sub, value[k], env):
                    return False
            else:
                if default is not None:
                    env[k] = (lambda d=default, e=dict(env): d.eval(e)) if lazy else default.eval(env)
                else:
                    return False
        return True

    # === Splat ===
    elif isinstance(pattern, SplatPattern):
        if pattern.beanstalk and isinstance(value, list):
            for j in range(1, len(value) + 1):
                env[pattern.var] = value[:j]
                run_beanstalk_hooks(pattern.var, env[pattern.var])
            return True
        env[pattern.var] = value
        return True

    # Unknown pattern
    return False

# -----------------------------
# Codegen / Evaluator + VM
# -----------------------------

def run_task(task, env):
    """
    Run a Task node directly.
    """
    for stmt in task.body:
        stmt.eval(env)
    return env


def run_program(ast):
    """
    Entry point: find and run 'task main'.
    """
    env = {}
    for node in ast:
        if isinstance(node, Task) and node.name == "main":
            run_task(node, env)
    return env


# === Expression Evaluator Helpers ===
def gen_expr(node, env):
    """
    Evaluate an expression node (Push, Var).
    """
    if isinstance(node, Push):
        return node.value
    elif isinstance(node, Var):
        v = env[node.name]
        if callable(v):   # lazy binding
            v = v()
            env[node.name] = v
        return v
    else:
        raise RuntimeError(f"Unknown expression node: {node}")


# === Statement Evaluators ===
class VarDecl(Node):
    def eval(self, env):
        env[self.name] = gen_expr(self.expr, env)
        return env[self.name]


class Say(Node):
    def eval(self, env):
        val = gen_expr(self.expr, env)
        print(val)
        return val


class Chart(Node):
    def eval(self, env):
        val = gen_expr(self.expr, env)
        if self.tree:
            print("Beanstalk Chart (tree):")
            print_tree(val)
        else:
            print("Beanstalk Chart (dash):")
            if isinstance(val, list):
                line = ""
                for i, v in enumerate(val):
                    line += str(v)
                    if i < len(val) - 1:
                        line += "—"
                    print(line)
            else:
                print(val)
        return val


class Match(Node):
    def eval(self, env):
        value = gen_expr(self.expr, env)
        for case in self.cases:
            local_env = dict(env)
            if case.pattern.match(value, local_env):
                for stmt in case.block:
                    stmt.eval(local_env)
                env.update(local_env)
                return True
        return False

# -----------------------------
# Stacker Virtual Machine (VM)
# -----------------------------

class VM:
    """
    The Stacker Virtual Machine.
    Holds the environment, tasks, and controls execution.
    """

    def __init__(self, ast):
        # Program AST (list of Task nodes)
        self.ast = ast
        # Global environment (variables, functions, bindings)
        self.env = {}
        # Task registry for direct lookup
        self.tasks = {node.name: node for node in ast if isinstance(node, Task)}

    # -------------------------
    # Task Management
    # -------------------------
    def run_task(self, name="main"):
        """
        Run a named task (default: main).
        """
        if name not in self.tasks:
            raise RuntimeError(f"Task '{name}' not found")
        task = self.tasks[name]
        for stmt in task.body:
            stmt.eval(self.env)
        return self.env

    def run_all(self):
        """
        Run all tasks in order of appearance.
        """
        for name in self.tasks:
            self.run_task(name)
        return self.env

    # -------------------------
    # Variable Management
    # -------------------------
    def get(self, varname):
        if varname not in self.env:
            raise RuntimeError(f"Variable '{varname}' not defined")
        val = self.env[varname]
        if callable(val):   # lazy binding support
            val = val()
            self.env[varname] = val
        return val

    def set(self, varname, value):
        self.env[varname] = value
        return value

    # -------------------------
    # I/O Helpers
    # -------------------------
    def say(self, value):
        """
        Print a value (with lazy evaluation support).
        """
        if callable(value):
            value = value()
        print(value)
        return value

    def chart(self, value, tree=False):
        """
        Print a value as a chart (dash or tree mode).
        """
        if tree:
            print("Beanstalk Chart (tree):")
            print_tree(value)
        else:
            print("Beanstalk Chart (dash):")
            if isinstance(value, list):
                line = ""
                for i, v in enumerate(value):
                    line += str(v)
                    if i < len(value) - 1:
                        line += "—"
                    print(line)
            else:
                print(value)
        return value

    # -------------------------
    # Match Execution
    # -------------------------
    def run_match(self, expr_node, cases):
        """
        Execute a match statement in VM context.
        """
        value = expr_node.eval(self.env)
        for case in cases:
            local_env = dict(self.env)
            if case.pattern.match(value, local_env):
                for stmt in case.block:
                    stmt.eval(local_env)
                self.env.update(local_env)
                return True
        return False

    # -----------------------------
# Driver (CLI Entrypoint)
# -----------------------------
import sys

def compile_and_run_file(filename):
    """
    Load a .stk source file, compile it into AST, and run it immediately.
    """
    with open(filename, "r") as f:
        src = f.read()

    # Step 1: Lex
    tokens = tokenize(src)

    # Step 2: Parse
    ast = parse(tokens)

    # Step 3: Run in VM
    vm = VM(ast)
    vm.run_task("main")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stacker.py program.stk")
        sys.exit(0)

    compile_and_run_file(sys.argv[1])
