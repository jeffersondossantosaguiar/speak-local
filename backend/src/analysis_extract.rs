/// Incremental extraction of complete JSON `{...}` objects from a streaming
/// LLM output, so the UI can render each error box as soon as the model
/// finishes it instead of waiting for the whole document.
///
/// The scanner is string-aware (quotes/escapes are honored), so braces inside
/// the error text or explanation never break an object boundary. Every fully
/// closed `{...}` is handed to the caller as a raw slice; the caller decides
/// whether it parses into an error (the enclosing document object, for
/// example, does not). This keeps the extractor independent of the analysis
/// schema.
#[derive(Default)]
pub struct ErrorObjectExtractor {
    buf: String,
    /// Byte position the forward scan resumes from.
    pos: usize,
    in_string: bool,
    escaped: bool,
    /// Opened delimiters (`{`/`[`) with their byte index, LIFO.
    stack: Vec<(u8, usize)>,
}

impl ErrorObjectExtractor {
    /// Feed the next chunk of JSON text. Every complete `{...}` discovered is
    /// passed to `on_object` as its raw text slice.
    pub fn feed(&mut self, delta: &str, on_object: &mut dyn FnMut(&str)) {
        self.buf.push_str(delta);

        let tail_start = self.pos;
        for (rel, ch) in self.buf[tail_start..].char_indices() {
            let abs = tail_start + rel;
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            match ch {
                '"' => self.in_string = true,
                '{' => self.stack.push((b'{', abs)),
                '[' => self.stack.push((b'[', abs)),
                '}' => {
                    if let Some((op, start)) = self.stack.pop() {
                        if op == b'{' {
                            let slice = &self.buf[start..=abs];
                            on_object(slice);
                        }
                    }
                }
                ']' => {
                    // Tolerate a stray closer (JSON is still being generated).
                    self.stack.pop();
                }
                _ => {}
            }
        }
        self.pos = self.buf.len();
    }
}

/// Convenience: extract every complete JSON object from a full string.
pub fn extract_objects(content: &str) -> Vec<String> {
    let mut extractor = ErrorObjectExtractor::default();
    let mut out = Vec::new();
    extractor.feed(content, &mut |slice| {
        if out.iter().all(|s: &String| s != slice) {
            out.push(slice.to_string());
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completed_objects(deltas: &[&str]) -> Vec<String> {
        let mut extractor = ErrorObjectExtractor::default();
        let mut out = Vec::new();
        for d in deltas {
            extractor.feed(d, &mut |slice| out.push(slice.to_string()));
        }
        out
    }

    #[test]
    fn emits_each_object_when_closed() {
        let doc = r#"{"cefr_level":"B1","errors":[{"text":"he go","suggestion":"he goes","category":"grammar"},{"text":"discloyer","suggestion":"disclosure"}],"cefr_justification":"x"}"#;
        // Arbitrary small chunks, mimicking token streaming.
        let mut deltas = Vec::new();
        let mut rest = doc;
        while !rest.is_empty() {
            let cut = if rest.len() > 7 { 5 } else { rest.len() };
            let (head, tail) = rest.split_at(cut);
            deltas.push(head);
            rest = tail;
        }

        let objs = completed_objects(&deltas);
        // The two error objects must appear individually, in order; the
        // enclosing document object closes last.
        assert!(objs.len() >= 3, "got {objs:?}");
        assert!(objs[0].contains("he go"));
        assert!(objs[1].contains("discloyer"));
    }

    #[test]
    fn single_delta_whole_document_yields_objects_plus_document() {
        let doc = r#"{"errors":[{"text":"a","suggestion":"b","category":"grammar","criticality":3,"context":"","explanation":""}]}"#;
        let objs = extract_objects(doc);
        // The error object plus the enclosing document object ("text" absent
        // at the top level — callers decide what parses as an error).
        assert_eq!(objs.len(), 2);
        assert!(objs[0].contains("\"a\""));
    }

    #[test]
    fn braces_inside_strings_do_not_split_objects() {
        let doc = r#"{"errors":[{"text":"use {braces} here","suggestion":"use braces here","category":"grammar"}]}"#;
        let objs = extract_objects(doc);
        assert!(objs.iter().any(|o| o.contains("use {braces} here")));
    }

    #[test]
    fn incomplete_document_emits_nothing_until_closed() {
        let doc = r#"{"errors":[{"text":"partia"#;
        let objs = extract_objects(doc);
        assert!(objs.is_empty());
    }
}