async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("Deploying contracts with the account:", deployer.address);
  console.log("Account balance:", (await deployer.getBalance()).toString());

  // Example: Deploy Governor contract (or replace with your own)
  // Make sure to adjust constructor params as needed!
  const Governor = await ethers.getContractFactory("RememberMeGovernor");
  const governor = await Governor.deploy(/* constructor args, e.g., token address */);
  await governor.deployed();

  console.log("Governor contract deployed to:", governor.address);

  // If you have multiple contracts:
  // const Token = await ethers.getContractFactory("MyToken");
  // const token = await Token.deploy();
  // await token.deployed();
  // console.log("Token deployed to:", token.address);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
