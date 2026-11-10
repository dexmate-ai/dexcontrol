// Copyright (C) 2026 Dexmate Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/* A minimal, dependency-free JSON reader for the examples.
 *
 * The wrapper returns structured results (health, diagnostics snapshots,
 * firmware replies, NTP queries) as JSON text; this parses that text into a
 * small value type. Lookups never throw: a missing key or index reads as a
 * null value, so `snapshot["monitoring"]["estop"]["source"].as_string()` is
 * safe however incomplete the snapshot is. Only `parse` throws, on malformed
 * input. Applications should use their own JSON library; this exists so the
 * examples have no dependencies.
 */

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace examples {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;

    static Json parse(const std::string &text) {
        if (text.size() > 16 * 1024 * 1024) throw std::runtime_error("JSON input exceeds 16 MiB");
        Parser parser{text, 0};
        Json value = parser.value();
        parser.skip_space();
        if (parser.at != text.size()) parser.error("trailing characters");
        return value;
    }

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::Null; }
    bool is_bool() const noexcept { return type_ == Type::Bool; }
    bool is_number() const noexcept { return type_ == Type::Number; }
    bool is_string() const noexcept { return type_ == Type::String; }
    bool is_array() const noexcept { return type_ == Type::Array; }
    bool is_object() const noexcept { return type_ == Type::Object; }

    bool as_bool(bool fallback = false) const noexcept { return is_bool() ? bool_ : fallback; }
    double as_number(double fallback = 0.0) const noexcept { return is_number() ? number_ : fallback; }
    const std::string &as_string(const std::string &fallback = empty_string()) const noexcept {
        return is_string() ? string_ : fallback;
    }
    /* A scalar rendered as text: strings verbatim, numbers compactly. */
    std::string text(const std::string &fallback = "") const {
        switch (type_) {
        case Type::Null: return fallback;
        case Type::Bool: return bool_ ? "true" : "false";
        case Type::Number: return number_text(number_);
        case Type::String: return string_;
        default: return dump(0);
        }
    }

    size_t size() const noexcept {
        return is_array() ? array_.size() : is_object() ? object_.size() : 0;
    }
    bool contains(const std::string &key) const noexcept {
        return is_object() && object_.count(key) != 0;
    }
    const Json &operator[](const std::string &key) const noexcept {
        if (!is_object()) return null_value();
        const auto found = object_.find(key);
        return found == object_.end() ? null_value() : found->second;
    }
    const Json &operator[](const char *key) const noexcept { return (*this)[std::string(key)]; }
    const Json &operator[](size_t index) const noexcept {
        return is_array() && index < array_.size() ? array_[index] : null_value();
    }
    const Json &operator[](int index) const noexcept {
        return index < 0 ? null_value() : (*this)[static_cast<size_t>(index)];
    }
    const std::vector<Json> &items() const noexcept {
        static const std::vector<Json> none;
        return is_array() ? array_ : none;
    }
    const std::map<std::string, Json> &members() const noexcept {
        static const std::map<std::string, Json> none;
        return is_object() ? object_ : none;
    }

    /* Pretty-printed with two-space indentation. */
    std::string dump(int indent = 2) const {
        std::ostringstream out;
        write(out, indent, 0);
        return out.str();
    }

private:
    static const Json &null_value() {
        static const Json value;
        return value;
    }
    static const std::string &empty_string() {
        static const std::string value;
        return value;
    }
    static std::string number_text(double value) {
        if (!std::isfinite(value)) return "null";  // JSON has no inf/nan
        std::ostringstream out;
        out << std::setprecision(15) << value;
        return out.str();
    }
    static void write_string(std::ostream &out, const std::string &value) {
        out << '"';
        for (const unsigned char c : value) {
            switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << c;
                }
            }
        }
        out << '"';
    }
    void write(std::ostream &out, int indent, int depth) const {
        const std::string pad(static_cast<size_t>(indent * (depth + 1)), ' ');
        const std::string close(static_cast<size_t>(indent * depth), ' ');
        const char *newline = indent > 0 ? "\n" : "";
        switch (type_) {
        case Type::Null: out << "null"; break;
        case Type::Bool: out << (bool_ ? "true" : "false"); break;
        case Type::Number: out << number_text(number_); break;
        case Type::String: write_string(out, string_); break;
        case Type::Array:
            if (array_.empty()) { out << "[]"; break; }
            out << "[" << newline;
            for (size_t i = 0; i < array_.size(); ++i) {
                out << (indent > 0 ? pad : "");
                array_[i].write(out, indent, depth + 1);
                out << (i + 1 < array_.size() ? "," : "") << newline;
            }
            out << (indent > 0 ? close : "") << "]";
            break;
        case Type::Object:
            if (object_.empty()) { out << "{}"; break; }
            out << "{" << newline;
            {
                size_t i = 0;
                for (const auto &[key, value] : object_) {
                    out << (indent > 0 ? pad : "");
                    write_string(out, key);
                    out << ": ";
                    value.write(out, indent, depth + 1);
                    out << (++i < object_.size() ? "," : "") << newline;
                }
            }
            out << (indent > 0 ? close : "") << "}";
            break;
        }
    }

    struct Parser {
        const std::string &text;
        size_t at;
        size_t depth = 0;

        [[noreturn]] void error(const std::string &what) const {
            throw std::runtime_error("JSON parse error at offset " + std::to_string(at) + ": " + what);
        }
        void skip_space() {
            while (at < text.size() && std::isspace(static_cast<unsigned char>(text[at]))) ++at;
        }
        char peek() {
            skip_space();
            if (at >= text.size()) error("unexpected end of input");
            return text[at];
        }
        void expect(char c) {
            if (peek() != c) error(std::string("expected '") + c + "'");
            ++at;
        }
        bool consume(const char *literal) {
            const size_t length = std::char_traits<char>::length(literal);
            if (text.compare(at, length, literal) != 0) return false;
            at += length;
            return true;
        }
        Json value() {
            if (depth >= 128) error("nesting limit exceeded");
            struct Depth { size_t &value; ~Depth() { --value; } } guard{depth};
            ++depth;
            const char c = peek();
            if (c == '{') return object();
            if (c == '[') return array();
            if (c == '"') { Json v; v.type_ = Type::String; v.string_ = string(); return v; }
            if (consume("true")) { Json v; v.type_ = Type::Bool; v.bool_ = true; return v; }
            if (consume("false")) { Json v; v.type_ = Type::Bool; v.bool_ = false; return v; }
            if (consume("null")) return Json();
            if (c == '-' || (c >= '0' && c <= '9')) return number();
            error("unexpected character");
        }
        Json number() {
            // RFC 8259: an optional minus, an integer with no leading zero,
            // an optional fraction and exponent, each with at least one digit.
            const size_t start = at;
            if (text[at] == '-') ++at;
            if (at >= text.size() || !std::isdigit(static_cast<unsigned char>(text[at]))) {
                error("expected a digit");
            }
            if (text[at] == '0' && at + 1 < text.size() &&
                std::isdigit(static_cast<unsigned char>(text[at + 1]))) {
                error("leading zeros are not allowed");
            }
            while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at]))) ++at;
            if (at < text.size() && text[at] == '.') {
                ++at;
                if (at >= text.size() || !std::isdigit(static_cast<unsigned char>(text[at]))) {
                    error("expected a digit after '.'");
                }
                while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at]))) ++at;
            }
            if (at < text.size() && (text[at] == 'e' || text[at] == 'E')) {
                ++at;
                if (at < text.size() && (text[at] == '+' || text[at] == '-')) ++at;
                if (at >= text.size() || !std::isdigit(static_cast<unsigned char>(text[at]))) {
                    error("expected a digit in the exponent");
                }
                while (at < text.size() && std::isdigit(static_cast<unsigned char>(text[at]))) ++at;
            }
            Json v;
            v.type_ = Type::Number;
            v.number_ = std::strtod(text.c_str() + start, nullptr);
            if (!std::isfinite(v.number_)) error("number is out of range");
            return v;
        }
        static void append_utf8(std::string &out, unsigned code) {
            if (code < 0x80) {
                out += static_cast<char>(code);
            } else if (code < 0x800) {
                out += static_cast<char>(0xC0 | (code >> 6));
                out += static_cast<char>(0x80 | (code & 0x3F));
            } else if (code < 0x10000) {
                out += static_cast<char>(0xE0 | (code >> 12));
                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (code & 0x3F));
            } else {
                out += static_cast<char>(0xF0 | (code >> 18));
                out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                out += static_cast<char>(0x80 | (code & 0x3F));
            }
        }
        /* Reads the four hex digits of a \u escape at `at`. */
        unsigned hex4() {
            if (at + 4 > text.size()) error("short \\u escape");
            for (size_t i = 0; i < 4; ++i) {
                if (!std::isxdigit(static_cast<unsigned char>(text[at + i]))) error("bad \\u escape");
            }
            const unsigned code =
                static_cast<unsigned>(std::strtoul(text.substr(at, 4).c_str(), nullptr, 16));
            at += 4;
            return code;
        }
        std::string string() {
            expect('"');
            std::string out;
            while (true) {
                if (at >= text.size()) error("unterminated string");
                const char c = text[at++];
                if (c == '"') return out;
                if (static_cast<unsigned char>(c) < 0x20) error("unescaped control character");
                if (c != '\\') { out += c; continue; }
                if (at >= text.size()) error("unterminated escape");
                const char e = text[at++];
                switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    unsigned code = hex4();
                    if (code >= 0xD800 && code <= 0xDBFF) {
                        // A high surrogate must be followed by a low one;
                        // together they encode one code point above U+FFFF.
                        if (at + 6 <= text.size() && text[at] == '\\' && text[at + 1] == 'u') {
                            const size_t rewind = at;
                            at += 2;
                            const unsigned low = hex4();
                            if (low >= 0xDC00 && low <= 0xDFFF) {
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            } else {
                                at = rewind;
                                code = 0xFFFD;
                            }
                        } else {
                            code = 0xFFFD;
                        }
                    } else if (code >= 0xDC00 && code <= 0xDFFF) {
                        code = 0xFFFD;  // a lone low surrogate
                    }
                    append_utf8(out, code);
                    break;
                }
                default: error("unknown escape");
                }
            }
        }
        Json array() {
            expect('[');
            Json v;
            v.type_ = Type::Array;
            if (peek() == ']') { ++at; return v; }
            while (true) {
                v.array_.push_back(value());
                const char c = peek();
                ++at;
                if (c == ']') return v;
                if (c != ',') error("expected ',' or ']'");
            }
        }
        Json object() {
            expect('{');
            Json v;
            v.type_ = Type::Object;
            if (peek() == '}') { ++at; return v; }
            while (true) {
                std::string key = string();
                expect(':');
                v.object_[key] = value();
                const char c = peek();
                ++at;
                if (c == '}') return v;
                if (c != ',') error("expected ',' or '}'");
            }
        }
    };

    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;
};

}  // namespace examples
