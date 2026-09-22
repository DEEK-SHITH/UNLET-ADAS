export default function AnimatedBackground() {
  return (
    <>
      <div
        className="bg-orb"
        style={{
          top: '-10%',
          left: '-10%',
          width: '45vw',
          height: '45vw',
          background: 'radial-gradient(circle, #5eead4, transparent 70%)',
          animation: 'drift-a 22s ease-in-out infinite',
        }}
      />
      <div
        className="bg-orb"
        style={{
          top: '20%',
          right: '-15%',
          width: '40vw',
          height: '40vw',
          background: 'radial-gradient(circle, #818cf8, transparent 70%)',
          animation: 'drift-b 26s ease-in-out infinite',
        }}
      />
      <div
        className="bg-orb"
        style={{
          bottom: '-15%',
          left: '20%',
          width: '38vw',
          height: '38vw',
          background: 'radial-gradient(circle, #22d3ee, transparent 70%)',
          animation: 'drift-c 30s ease-in-out infinite',
        }}
      />
    </>
  );
}
