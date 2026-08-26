import re

# Only spans that are actually tag-shaped.  A bare "<" is ordinary prose in a
# model answer — "mean 0.709 < 0.712", a generic in a sentence, a stray
# comparison — and a pattern of <[^>]+> pairs that "<" with the next ">" it can
# find, which may be a closing wrapper tag thousands of characters later.  The
# whole answer between the two then disappears, silently and only for answers
# that happen to contain the character.  So the name has to look like a name:
# "<" followed by whitespace or a digit is left alone, and no tag may span a
# second "<".  The alternatives cover model special tokens (<|im_end|>) and
# comments/doctypes, which are tags for this purpose but not named ones.
_TAG_RE = re.compile(
    r"""<
        (?:
            /?[A-Za-z][A-Za-z0-9:._-]*      # <b>  </div>  <answer>  <br/>
            (?:\s[^<>]*)?                   #   attributes
            /?
          | \|[^<>|]{0,64}\|                # <|im_end|>, <|endoftext|>
          | ![^<>]{0,400}                   # <!-- comment -->, <!DOCTYPE ...>
        )
    >""",
    re.VERBOSE,
)

# Separates the two halves of an agent instruction.
#
# Everything before it is identical from one request to the next — the agent's
# role, the research procedure, the standing rules — and so belongs in the
# system message, ahead of the session history, where a server with prefix
# caching sees the same bytes on every request from every session and skips
# prefilling them. Everything after it is what this session and this task
# supply: the date, the working directory, the file server, the task itself.
#
# Deliberately not a tag: remove_tags() runs over model output and would eat an
# HTML-comment sentinel if one ever came back through.
INSTRUCTION_SPLIT = "\n[[onit:session-context]]\n"

# Opens any block the model must read but must not answer.  Without it a weak
# model treats a reference section as something to comply with and replies
# "Working directory confirmed: <uuid>" — the data_path basename read back —
# instead of doing the task.  Shared because the same marker has to open the
# context block in the prompt builder and the resume note in RunState: the two
# render into one instruction a few hundred tokens apart, and a partial reword
# ships two contradictory versions of the same rule.
REFERENCE_ONLY = "Reference only. Do not repeat or acknowledge this section."


def split_instruction(instruction: str) -> tuple[str, str]:
    """Split an instruction into its (static, volatile) halves.

    An instruction with no sentinel is treated as entirely volatile, so a
    caller that assembled one by hand still gets a correct message list — it
    simply forfeits the shared prefix.  Every instruction this package builds
    carries the sentinel, custom templates included.
    """
    static, sentinel, volatile = (instruction or "").partition(INSTRUCTION_SPLIT)
    if not sentinel:
        return "", instruction
    return static, volatile

def remove_tags(text: str) -> str:
    """
    Remove all HTML/XML tags from text.

    Args:
        text: The text to process
    Returns:
        The text with all tags removed
    """
    if not text:
        return text
    return _TAG_RE.sub('', text)

def text_between_tags(text: str, tag: str) -> tuple[bool, str]:
    """
    Extract text between <tag> and </tag> tags.
    
    Args:
        text: The text to search in
        tag: The tag name without angle brackets
        
    Returns:
        tuple: (is_full_match, extracted_text)
            is_full_match: True if text starts with <tag> and ends with </tag>
            extracted_text: The text between tags or original text if tags not found
    """
    if not text or not tag:
        return False, text
        
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    # Check if text is fully wrapped in the specified tags
    is_full_match = text.startswith(start_tag) and text.endswith(end_tag)
    
    # Find the last occurrence of the start and end tags
    start_index = text.rfind(start_tag)
    if start_index == -1:
        return False, text
        
    end_index = text.rfind(end_tag)
    if end_index == -1 or end_index <= start_index:
        return False, text
    
    # Extract the text between the tags
    extracted_text = text[start_index + len(start_tag):end_index].strip()
    return is_full_match, extracted_text