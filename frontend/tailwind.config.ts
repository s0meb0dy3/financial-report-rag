import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#f7f4ee",
        ink: "#171717",
        muted: "#6f6a60",
        line: "#dfd8cc",
        moss: "#1f7a57",
        brass: "#a26f24",
      },
      fontFamily: {
        sans: [
          "Avenir Next",
          "Gill Sans",
          "PingFang SC",
          "Hiragino Sans GB",
          "Microsoft YaHei",
          "sans-serif",
        ],
        serif: ["Iowan Old Style", "Songti SC", "STSong", "serif"],
      },
      boxShadow: {
        soft: "0 18px 60px rgba(30, 28, 24, 0.10)",
      },
      keyframes: {
        rise: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        fade: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
      },
      animation: {
        rise: "rise 480ms cubic-bezier(.2,.8,.2,1) both",
        fade: "fade 360ms ease both",
      },
    },
  },
  plugins: [],
} satisfies Config;
