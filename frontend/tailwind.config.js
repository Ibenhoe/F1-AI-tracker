/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      container: {
        center: true,
        padding: "1.25rem",
        screens: {
          "2xl": "1280px",
        },
      },
    },
  },
  plugins: [],
};
