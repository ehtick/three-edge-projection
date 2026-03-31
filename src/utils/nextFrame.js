export const nextFrame = () => new Promise( resolve => {

	let rafHandle;
	let timeoutHandle;
	const cb = () => {

		cancelAnimationFrame( rafHandle );
		clearTimeout( timeoutHandle );
		resolve();

	};

	rafHandle = requestAnimationFrame( cb );
	timeoutHandle = setTimeout( cb, 16 );

} );
