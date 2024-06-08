const sr = ScrollReveal({
    origin: "top",
    distance: "50px",
    duration: 1500,
    delay: 200,
    // reset: true
});

sr.reveal(
    ` .title, .sectionTitleBar, .cards, .timeline, .contact-text, .input-container, .floaterHeader, .imageBannerAnimation, .projectImageAnimation, .projectAnimation, .backBtn`
);
sr.reveal(
    `.links, .skills, .card_animation, .timelineBody_animation, .textBannerAnimation, .techLogosAnimation, .websiteImageAnimation`, {
        interval: 100,
    }
);
sr.reveal(`.heroImage`, { origin: "left" });
sr.reveal(`.heroText`, { origin: "right" });

var rellax = new Rellax(".parallax");