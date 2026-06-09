(function () {
  const deck = document.getElementById('deck');
  const slides = Array.from(document.querySelectorAll('.slide'));
  const indicator = document.getElementById('pageIndicator');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const overviewBtn = document.getElementById('overviewBtn');
  const notesBtn = document.getElementById('notesBtn');
  const notesPanel = document.getElementById('speakerNotes');
  let index = Number(new URLSearchParams(location.search).get('slide') || location.hash.replace('#', '')) || 0;
  let notesVisible = false;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function scaleDeck() {
    if (document.body.classList.contains('overview')) return;
    const scale = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
    deck.style.transform = `translate(-50%, -50%) scale(${scale})`;
  }

  function setSlide(nextIndex) {
    index = clamp(nextIndex, 0, slides.length - 1);
    slides.forEach((slide, i) => slide.classList.toggle('active', i === index));
    indicator.textContent = `${index + 1} / ${slides.length}`;
    location.hash = String(index);
    updateNotes();
  }

  function updateNotes() {
    const notes = slides[index].querySelector('.notes');
    notesPanel.textContent = notes ? notes.textContent.trim() : '本页没有演讲者备注。';
    notesPanel.hidden = !notesVisible;
  }

  function toggleOverview() {
    const entering = !document.body.classList.contains('overview');
    document.body.classList.toggle('overview', entering);
    if (entering) {
      slides.forEach(slide => slide.classList.remove('active'));
      deck.style.transform = '';
    } else {
      setSlide(index);
      scaleDeck();
    }
  }

  prevBtn.addEventListener('click', () => setSlide(index - 1));
  nextBtn.addEventListener('click', () => setSlide(index + 1));
  overviewBtn.addEventListener('click', toggleOverview);
  notesBtn.addEventListener('click', () => {
    notesVisible = !notesVisible;
    updateNotes();
  });

  slides.forEach((slide, i) => {
    slide.addEventListener('click', () => {
      if (!document.body.classList.contains('overview')) return;
      document.body.classList.remove('overview');
      setSlide(i);
      scaleDeck();
    });
  });

  window.addEventListener('resize', scaleDeck);
  window.addEventListener('keydown', event => {
    if (event.key === 'ArrowRight' || event.key === 'PageDown' || event.key === ' ') {
      event.preventDefault();
      if (!document.body.classList.contains('overview')) setSlide(index + 1);
    }
    if (event.key === 'ArrowLeft' || event.key === 'PageUp') {
      event.preventDefault();
      if (!document.body.classList.contains('overview')) setSlide(index - 1);
    }
    if (event.key.toLowerCase() === 'o') {
      event.preventDefault();
      toggleOverview();
    }
    if (event.key.toLowerCase() === 'n') {
      event.preventDefault();
      notesVisible = !notesVisible;
      updateNotes();
    }
    if (event.key === 'Escape' && document.body.classList.contains('overview')) {
      document.body.classList.remove('overview');
      setSlide(index);
      scaleDeck();
    }
  });

  index = clamp(index, 0, slides.length - 1);
  setSlide(index);
  scaleDeck();
})();
